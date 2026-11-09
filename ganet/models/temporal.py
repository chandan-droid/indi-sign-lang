"""Stage 2 (temporal modeling): Conv1D block -> [Self-Attention -> BiLSTM] x2 -> classifier (paper Fig. 2).

num_pool = number of temporal 2x poolings in the Conv1D block (VAC uses 2). For short clips use 0 or 1,
otherwise the output sequence can be shorter than the label and CTC becomes infeasible."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .heads import length_mask


class SelfAttnBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, pad_mask):
        y, _ = self.attn(x, x, x, key_padding_mask=pad_mask, need_weights=False)
        return self.norm(x + y)


class TemporalStage(nn.Module):
    def __init__(self, dim, vocab_size, num_pool=2, use_attn=True, lstm_layers=2, dropout=0.3, heads=8):
        super().__init__()
        self.num_pool = num_pool
        blocks = []
        for _ in range(max(1, num_pool)):
            blocks += [nn.Conv1d(dim, dim, 5, padding=2), nn.BatchNorm1d(dim), nn.ReLU(inplace=True)]
            if num_pool > 0:
                blocks.append(nn.MaxPool1d(2))
        self.conv = nn.Sequential(*blocks)
        self.attn = nn.ModuleList([SelfAttnBlock(dim, heads) if use_attn else nn.Identity()
                                   for _ in range(lstm_layers)])
        self.lstm = nn.ModuleList([nn.LSTM(dim, dim // 2, batch_first=True, bidirectional=True)
                                   for _ in range(lstm_layers)])
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, vocab_size)
        self.use_attn = use_attn

    def out_lengths(self, lengths):
        return (lengths // (2 ** self.num_pool)).clamp_min(1)

    def forward(self, feats, lengths):
        """feats [B, T, D], lengths [B] -> dict(conv_feat, seq_logits, out_len)."""
        x = self.conv(feats.transpose(1, 2)).transpose(1, 2)       # [B, L, D]
        L = x.size(1)
        lens = self.out_lengths(lengths).clamp_max(L)
        conv_feat = x
        pad_mask = ~length_mask(lens, L)
        for attn, lstm in zip(self.attn, self.lstm):
            if self.use_attn:
                x = attn(x, pad_mask)
            packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
            x, _ = lstm(packed)
            x, _ = pad_packed_sequence(x, batch_first=True, total_length=L)
            x = self.drop(x)
        return dict(conv_feat=conv_feat, seq_logits=self.classifier(x), out_len=lens, bilstm_feat=x)
