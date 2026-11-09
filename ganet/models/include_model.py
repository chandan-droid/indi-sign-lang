import torch.nn as nn
from .visual import VisualStage1


class IncludeClassifier(nn.Module):
    """Stage 1 + throw-away classification head, for isolated-sign pre-training on INCLUDE."""

    def __init__(self, num_classes, gcmm=True, gam=True, imagenet=True, temporal_mix=True, dropout=0.3):
        super().__init__()
        self.visual = VisualStage1(gcmm, gam, imagenet, temporal_mix)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(self.visual.out_dim, num_classes)

    def forward(self, x):                                          # x [B, T, 3, H, W]
        return self.fc(self.drop(self.visual(x).mean(1)))
