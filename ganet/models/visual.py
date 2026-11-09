"""Stage 1 (visual modeling): SMM (ResNet-18 stages) + GCMM branch + GAM fusion after each stage.

This exact module is shared by the INCLUDE pre-training model and the ISL-CSLTR model, so weights
transfer with identical key names (state_dict prefix 'visual.')."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .gcmm import GCMM
from .gam import GAM


def _resnet18(imagenet):
    try:
        return torchvision.models.resnet18(weights='IMAGENET1K_V1' if imagenet else None)
    except TypeError:                                       # old torchvision
        return torchvision.models.resnet18(pretrained=imagenet)


class VisualStage1(nn.Module):
    smm_channels = (64, 128, 256, 512)

    def __init__(self, use_gcmm=True, use_gam=True, imagenet=True, temporal_mix=True):
        super().__init__()
        assert not (use_gam and not use_gcmm), 'GAM needs the GCMM branch'
        r = _resnet18(imagenet)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layers = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])
        self.out_dim = 512
        self.gcmm = GCMM(temporal_mix) if use_gcmm else None
        self.gams = (nn.ModuleList([GAM(c, g) for c, g in zip(self.smm_channels, GCMM.channels)])
                     if use_gam else None)

    def forward(self, x):
        """x: [B, T, 3, H, W] -> frame features [B, T, 512]."""
        B, T = x.shape[:2]
        gfeats = self.gcmm(x.transpose(1, 2).contiguous()) if self.gcmm is not None else None
        m = self.stem(x.flatten(0, 1))
        for i, layer in enumerate(self.layers):
            m = layer(m)
            if self.gams is not None:
                C, H, W = m.shape[1:]
                m5 = m.reshape(B, T, C, H, W).transpose(1, 2).contiguous()      # [B, C, T, H, W]
                g = gfeats[i]
                if g.shape[-2:] != m5.shape[-2:]:
                    g = F.adaptive_avg_pool3d(g, (T, H, W))
                m5 = self.gams[i](m5, g)
                m = m5.transpose(1, 2).reshape(B * T, C, H, W)
        return m.mean((2, 3)).reshape(B, T, -1)
