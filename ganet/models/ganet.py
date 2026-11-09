import torch.nn as nn

from .visual import VisualStage1
from .temporal import TemporalStage
from .heads import GLPHead, VACHead


class GANet(nn.Module):
    """Two-stage GANet (RGB). Flags let you build the baseline, paper, and modified architectures."""

    def __init__(self, vocab_size, glp_bins=12, gcmm=True, gam=True, glp=True, glp_tap='temporal',
                 vac=False, vac_tap='visual', num_pool=2, attn=True, imagenet=True, temporal_mix=True):
        super().__init__()
        assert vac_tap in ('conv', 'visual')
        assert glp_tap in ('conv', 'visual', 'temporal')
        self.vac_tap = vac_tap
        self.glp_tap = glp_tap
        self.visual = VisualStage1(gcmm, gam, imagenet, temporal_mix)
        self.temporal = TemporalStage(self.visual.out_dim, vocab_size, num_pool, attn)
        self.glp = GLPHead(self.visual.out_dim, glp_bins) if glp else None
        self.vac = VACHead(self.visual.out_dim, vocab_size) if vac else None

    def forward(self, video, vlen):
        f = self.visual(video)                                     # [B, T, 512]
        t = self.temporal(f, vlen)
        out = dict(seq_logits=t['seq_logits'], out_len=t['out_len'])
        if self.glp is not None:
            if self.glp_tap == 'temporal':                         # BiLSTM layers (modified arch diagram)
                out['glp_logits'] = self.glp(t['bilstm_feat'], t['out_len'])
            elif self.glp_tap == 'conv':
                out['glp_logits'] = self.glp(t['conv_feat'], t['out_len'])
            else:                                                  # Stage 1 visual features
                out['glp_logits'] = self.glp(f, vlen)
        if self.vac is not None:
            if self.vac_tap == 'conv':                             # original VAC: head on the conv branch
                out['vac_logits'], out['vac_len'] = self.vac(t['conv_feat']), t['out_len']
            else:                                                  # tap directly after Stage 1 (visual modeling)
                out['vac_logits'], out['vac_len'] = self.vac(f), vlen
        return out


def build_model(mcfg, vocab_size):
    return GANet(vocab_size, glp_bins=mcfg.get('glp_bins', 12), gcmm=mcfg.get('gcmm', True),
                 gam=mcfg.get('gam', True), glp=mcfg.get('glp', True),
                 glp_tap=mcfg.get('glp_tap', 'temporal'),
                 vac=mcfg.get('vac', False),
                 vac_tap=mcfg.get('vac_tap', 'visual'),
                 num_pool=mcfg.get('num_pool', 2),
                 attn=mcfg.get('attn', True), imagenet=mcfg.get('imagenet', True),
                 temporal_mix=mcfg.get('temporal_mix', True))
