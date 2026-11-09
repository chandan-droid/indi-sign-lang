"""Global Context Modeling Module (GCMM): lightweight low-channel branch that scans all frames.

Paper: channels 16/32/64/128 across four stages, spatial kernel 1x3x3 (best in their Table 4).
Each stage downsamples space by 2 so its feature maps match the ResNet-18 stages exactly for a
224x224 input: 56, 28, 14, 7. The temporal 3x1x1 conv is an interpretation of the paper's remark
that temporal information is preserved alongside spatial modeling (set temporal_mix=False to drop it).
"""
import torch.nn as nn


class GCMMStage(nn.Module):
    def __init__(self, cin, cout, temporal_mix=True):
        super().__init__()
        layers = [nn.Conv3d(cin, cout, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
                  nn.BatchNorm3d(cout), nn.ReLU(inplace=True)]
        if temporal_mix:
            layers += [nn.Conv3d(cout, cout, (3, 1, 1), padding=(1, 0, 0), bias=False),
                       nn.BatchNorm3d(cout), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class GCMM(nn.Module):
    channels = (16, 32, 64, 128)

    def __init__(self, temporal_mix=True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(3, 16, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True))
        cs = (16,) + self.channels
        self.stages = nn.ModuleList([GCMMStage(cs[i], cs[i + 1], temporal_mix) for i in range(4)])

    def forward(self, x):
        """x: [B, 3, T, H, W] -> list of 4 feature maps [B, C_i, T, H_i, W_i]."""
        x = self.stem(x)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats
