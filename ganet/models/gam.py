"""Gloss-Aware Module (GAM) — paper-faithful implementation of Figure 4.

Architecture (Figure 4, Sec. 3.3):

  Space  (T×4C×H×W) ──► ST-Attention ──────────────────────────────► Space Output (T×4C×H×W)
                                                                               │
                              Cross Attention ◄── Global Output                │
                                    │                   │                      ▼
  Global (T×C×H×W)  ──► ST-Attention ──► Global Output   Conv Block ──► Output (T×C×H×W)

STAttention (STConv block, blue box in Fig. 4):
  Input T×C×H×W
  ├─ Temporal path: Pool(H,W) → Conv1D → Sigmoid → ⊙ (element-wise multiply)
  ├─ Spatial  path: Conv3D   → Sigmoid → ⊙
  └─ → DilatedCluster

DilatedCluster (right half of STConv, Fig. 4):
  ├─ T-Pool (avg H,W) → multi-scale dilated Conv1D, Wi-softmax fusion → broadcast [B,C,T,1,1]
  └─ S-Pool (avg T)   → multi-scale dilated Conv2D, Wi-softmax fusion → broadcast [B,C,1,H,W]
  → residual-add both to input

CrossAttention (orange block):
  Q from Space Output (spatially pooled → [B,T,C])
  K,V from Global Output (spatially pooled → [B,T,C])
  Multi-head attention over T axis → broadcast back to spatial dims

Conv Block: Conv3d + BN + ReLU (zero-init BN weight → identity at start of fine-tuning)

Note: The paper uses 'T×4C×H×W' for the Space branch (4× channels from SMM multi-resolution).
In this implementation c_main are the ResNet-18 layer channels {64,128,256,512} and c_glob are
the GCMM channels {16,32,64,128}. GAM projects c_glob → c_main via g_proj before CrossAttention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DilatedCluster: separate temporal (Conv1D) and spatial (Conv2D) paths
# ---------------------------------------------------------------------------

class DilatedCluster(nn.Module):
    """
    Paper Fig. 4 — right half of the blue STConv box.

    Temporal branch:  T-Pool (avg H,W) → dilated Conv1D×K → softmax-Wi fusion
    Spatial  branch:  S-Pool (avg T)   → dilated Conv2D×K → softmax-Wi fusion
    Both branches are broadcast back to [B, C, T, H, W] and added residually.
    """

    def __init__(self, c: int, dilations=(1, 2, 3)):
        super().__init__()
        # --- Temporal branch: Conv1D on [B, C, T] ---
        self.t_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(c, c, kernel_size=3, padding=d, dilation=d,
                          groups=c, bias=False),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
            )
            for d in dilations
        ])
        self.t_w = nn.Parameter(torch.zeros(len(dilations)))   # Wi for temporal

        # --- Spatial branch: Conv2D on [B, C, H, W] ---
        self.s_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, kernel_size=3, padding=d, dilation=d,
                          groups=c, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            )
            for d in dilations
        ])
        self.s_w = nn.Parameter(torch.zeros(len(dilations)))   # Wi for spatial

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T, H, W]"""
        # --- T-Pool → dilated Conv1D → expand ---
        t_in = x.mean(dim=(3, 4))                              # [B, C, T]
        t_wt = torch.softmax(self.t_w, dim=0)
        t_out = sum(w * b(t_in) for w, b in zip(t_wt, self.t_branches))   # [B, C, T]
        t_out = t_out[:, :, :, None, None].expand_as(x)       # [B, C, T, H, W]

        # --- S-Pool → dilated Conv2D → expand ---
        s_in = x.mean(dim=2)                                   # [B, C, H, W]
        s_wt = torch.softmax(self.s_w, dim=0)
        s_out = sum(w * b(s_in) for w, b in zip(s_wt, self.s_branches))   # [B, C, H, W]
        s_out = s_out[:, :, None, :, :].expand_as(x)          # [B, C, T, H, W]

        return x + t_out + s_out                               # residual fusion


# ---------------------------------------------------------------------------
# STAttention: temporal gate + spatial gate → DilatedCluster
# ---------------------------------------------------------------------------

class STAttention(nn.Module):
    """
    Paper Fig. 4 — STConv block (left half of blue box).

    Temporal path: Pool(H,W) → Conv1D → Sigmoid → ⊙  (broadcast over H,W)
    Spatial  path: Conv3D(1×3×3) → Sigmoid → ⊙
    → DilatedCluster
    """

    def __init__(self, c: int):
        super().__init__()
        # Temporal attention gate (depthwise Conv1D on pooled signal)
        self.temporal = nn.Sequential(
            nn.Conv1d(c, c, kernel_size=3, padding=1, groups=c, bias=True),
            nn.Sigmoid(),
        )
        # Spatial attention gate (1×3×3 so it's purely spatial, not temporal)
        self.spatial = nn.Sequential(
            nn.Conv3d(c, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=True),
            nn.Sigmoid(),
        )
        self.cluster = DilatedCluster(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T, H, W] → [B, C, T, H, W]"""
        # Temporal gate
        t_att = self.temporal(x.mean(dim=(3, 4)))              # [B, C, T]
        t_att = t_att[:, :, :, None, None]                     # broadcast to [B,C,T,1,1]

        # Spatial gate
        s_att = self.spatial(x)                                # [B, 1, T, H, W]

        return self.cluster(x * t_att * s_att)


# ---------------------------------------------------------------------------
# CrossAttention: Global Output → guides Space Output (over T-axis)
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """
    Paper Fig. 4 — orange Cross Attention block.

    Q from Space Output, K and V from Global Output.
    Attention is computed over the temporal axis T (after spatial avg-pool),
    keeping complexity O(T²) instead of O(T·H·W)² — practically: T≤160.
    The attended features are broadcast back to the full [B, C, T, H, W] shape.
    """

    def __init__(self, c: int, num_heads: int = 4):
        super().__init__()
        # Ensure divisibility; fall back to 1 head for very small channels
        while c % num_heads != 0 and num_heads > 1:
            num_heads //= 2
        self.num_heads = num_heads
        self.head_dim = c // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.proj = nn.Linear(c, c, bias=False)

    def forward(self, space: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
        """
        space: [B, C, T, H, W]  (Space Output after ST-Attention)
        glob:  [B, C, T, H, W]  (Global Output after ST-Attention + g_proj)
        → [B, C, T, H, W]
        """
        B, C, T, H, W = space.shape
        h, d = self.num_heads, self.head_dim

        # Spatial average-pool → [B, T, C] sequence for attention
        s_seq = space.mean(dim=(3, 4)).transpose(1, 2)         # [B, T, C]
        g_seq = glob.mean(dim=(3, 4)).transpose(1, 2)          # [B, T, C]

        Q = self.q(s_seq).view(B, T, h, d).transpose(1, 2)    # [B, h, T, d]
        K = self.k(g_seq).view(B, T, h, d).transpose(1, 2)    # [B, h, T, d]
        V = self.v(g_seq).view(B, T, h, d).transpose(1, 2)    # [B, h, T, d]

        attn = (Q @ K.transpose(-2, -1)) * self.scale         # [B, h, T, T]
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V)                                       # [B, h, T, d]
        out = out.transpose(1, 2).contiguous().view(B, T, C)   # [B, T, C]
        out = self.proj(out).transpose(1, 2)                   # [B, C, T]

        # Broadcast attention output back to spatial dims
        return out[:, :, :, None, None].expand_as(space)       # [B, C, T, H, W]


# ---------------------------------------------------------------------------
# GAM: full module (Figure 4 top-level)
# ---------------------------------------------------------------------------

class GAM(nn.Module):
    """
    Gloss-Aware Module — paper Figure 4.

    Flow:
      Space  → ST-Attention(main) ──────────────────────────────────► m_att
      Global → ST-Attention(glob) → g_proj(c_glob→c_main) → g_att   │
                                                               ↓      │
                                                      CrossAttention(m_att, g_att)
                                                               ↓
                                                          Conv Block (Conv3d+BN+ReLU)
                                                               ↓
                                              main + zero-init-BN(output) ← residual
    """

    def __init__(self, c_main: int, c_glob: int):
        super().__init__()
        self.g_att = STAttention(c_glob)                       # global branch ST-Attention
        self.m_att = STAttention(c_main)                       # space  branch ST-Attention
        self.g_proj = nn.Conv3d(c_glob, c_main, 1, bias=True)  # project c_glob → c_main

        self.cross_attn = CrossAttention(c_main, num_heads=4)  # orange Cross Attention block

        # Conv Block: Conv3d + BN + ReLU (shown as dashed-border box in legend)
        self.conv_block = nn.Sequential(
            nn.Conv3d(c_main, c_main, kernel_size=1, bias=False),
            nn.BatchNorm3d(c_main),
            nn.ReLU(inplace=True),
        )

        # Final BN — zero-weight-init so GAM starts as an identity mapping.
        # This prevents corrupting pre-trained backbone weights at the start of fine-tuning.
        self.out_bn = nn.BatchNorm3d(c_main)
        nn.init.zeros_(self.out_bn.weight)

    def forward(self, main: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
        """
        main: [B, C_main, T, H, W]  — SMM (Space branch)
        glob: [B, C_glob, T, H, W]  — GCMM (Global branch), same T,H,W after adaptive pool
        → [B, C_main, T, H, W]
        """
        # --- ST-Attention on both branches ---
        g = self.g_proj(self.g_att(glob))                      # [B, C_main, T, H, W]
        m = self.m_att(main)                                   # [B, C_main, T, H, W]

        # --- Cross Attention: Global guides Space ---
        ca = self.cross_attn(m, g)                             # [B, C_main, T, H, W]

        # --- Conv Block ---
        out = self.conv_block(ca)                              # [B, C_main, T, H, W]

        # --- Residual (Space Output = m + ConvBlock(CrossAttn)) ---
        return m + self.out_bn(out)
