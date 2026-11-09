"""Gloss-Aware Module (GAM).

Follows the paper's Fig. 4 and Sec. 3.3 description: an ST-Attention block (temporal + spatial attention,
then a parallel multi-scale dilated convolution cluster with learned softmax fusion weights) is applied to
both the global (GCMM) and the main (SMM) features; global features then guide the main features through a
channel gate (C-ATTN) and a spatiotemporal cross gate. This is an interpretation: the paper gives no code.
The last BatchNorm is zero-initialised, so GAM starts as an identity mapping and cannot damage
pre-trained backbone features at the beginning of fine-tuning.
"""
import torch
import torch.nn as nn


class DilatedCluster(nn.Module):
    def __init__(self, c, dilations=(1, 2, 3)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(c, c, (3, 1, 1), padding=(d, 0, 0), dilation=(d, 1, 1), bias=False),
                nn.BatchNorm3d(c), nn.ReLU(inplace=True)) for d in dilations])
        self.w = nn.Parameter(torch.zeros(len(dilations)))     # learned scale weights (softmax)

    def forward(self, x):
        w = torch.softmax(self.w, 0)
        return sum(wi * b(x) for wi, b in zip(w, self.branches))


class STAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.temporal = nn.Sequential(nn.Conv1d(c, c, 3, padding=1, groups=c), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv3d(c, 1, (1, 3, 3), padding=(0, 1, 1)), nn.Sigmoid())
        self.cluster = DilatedCluster(c)

    def forward(self, x):                                   # [B, C, T, H, W]
        t = self.temporal(x.mean((3, 4)))[:, :, :, None, None]
        s = self.spatial(x)
        return self.cluster(x * t * s)


class GAM(nn.Module):
    def __init__(self, c_main, c_glob):
        super().__init__()
        self.g_att = STAttention(c_glob)
        self.m_att = STAttention(c_main)
        self.g_proj = nn.Conv3d(c_glob, c_main, 1)
        self.chan = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Conv3d(c_main, c_main, 1), nn.Sigmoid())
        self.cross = nn.Sequential(nn.Conv3d(c_main, 1, 1), nn.Sigmoid())
        self.out = nn.Sequential(nn.Conv3d(c_main, c_main, 1, bias=False), nn.BatchNorm3d(c_main))
        nn.init.zeros_(self.out[1].weight)                  # identity at initialisation

    def forward(self, main, glob):
        """main: [B, Cm, T, H, W]; glob: [B, Cg, T, H, W] (same T, H, W) -> [B, Cm, T, H, W]."""
        g = self.g_proj(self.g_att(glob))
        m = self.m_att(main)
        gate = self.chan(g)                                 # global context -> channel gate
        cross = self.cross(g)                               # global context -> spatiotemporal gate
        return main + self.out(m * gate * (1.0 + cross) + g)
