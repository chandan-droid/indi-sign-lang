import torch
import torch.nn as nn


def length_mask(lengths, max_len):
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


class GLPHead(nn.Module):
    """Auxiliary gloss-length prediction: masked mean-pool of visual features -> simple MLP -> K bins.
    The paper (Table 4) found a simple MLP head better than an attention-based one."""

    def __init__(self, dim, num_bins, hidden=256, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(hidden, num_bins))

    def forward(self, feats, lengths):                      # feats [B, T, C]
        m = length_mask(lengths, feats.size(1)).unsqueeze(-1).to(feats.dtype)
        pooled = (feats * m).sum(1) / m.sum(1).clamp_min(1.0)
        return self.mlp(pooled)


class VACHead(nn.Module):
    """Frame-level visual classifier used for the auxiliary CTC (and optional distillation)."""

    def __init__(self, dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(dim, vocab_size)

    def forward(self, feats):
        return self.fc(feats)
