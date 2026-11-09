"""Continuous-sign dataset (ISL-CSLTR or any sentence-level set).

Manifest CSV columns: id,frame_dir,signer,text   (text = space-separated label words, e.g. the sentence)
Split JSON: {"train": [ids], "dev": [ids], "test": [ids]}; vocabulary is built from the train split only."""
import csv, json, re
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import list_frames, read_clip
from .augment import temporal_rescale, temporal_mask


def tokenize(text, lowercase=True):
    text = text.lower() if lowercase else text
    return re.sub(r'[^\w\s]', ' ', text).split()


class Vocab:
    def __init__(self, words):
        self.itos = ['<blank>'] + list(words)
        self.stoi = {w: i for i, w in enumerate(self.itos) if i > 0}

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        return [self.stoi[t] for t in toks if t in self.stoi]

    def decode(self, ids):
        return [self.itos[i] for i in ids]

    def save(self, path):
        json.dump(self.itos[1:], open(path, 'w'), ensure_ascii=False, indent=1)

    @staticmethod
    def load(path):
        return Vocab(json.load(open(path)))


def read_manifest(path):
    return {r['id']: r for r in csv.DictReader(open(path))}


def build_vocab(rows, train_ids, lowercase=True):
    words = sorted({w for i in train_ids for w in tokenize(rows[i]['text'], lowercase)})
    return Vocab(words)


def count_bin(n_words, num_bins):
    """Gloss count -> class index in [0, num_bins-1]; the last bin means 'num_bins or more'."""
    return min(max(n_words, 1), num_bins) - 1


class ContinuousSignDataset(Dataset):
    def __init__(self, rows, ids, vocab, train, dcfg, num_pool=2, glp_bins=12):
        self.rows, self.ids, self.vocab, self.train = rows, list(ids), vocab, train
        self.c, self.pool, self.bins = dcfg, 2 ** num_pool, glp_bins

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        r = self.rows[self.ids[i]]
        toks = tokenize(r['text'], self.c.get('lowercase', True))
        label = self.vocab.encode(toks)
        files = list_frames(r['frame_dir'])
        idx = np.arange(len(files))
        if self.train:
            if self.c.get('temporal_scale', True):
                idx = temporal_rescale(idx, *self.c.get('scale_range', [0.8, 1.2]))
            if self.c.get('mask_p', 0.0) > 0:
                repeats = sum(a == b for a, b in zip(label, label[1:]))
                need = (len(label) + repeats + 1) * self.pool        # keep CTC feasible after masking
                idx = temporal_mask(idx, self.c.get('mask_max_drop', 0.25),
                                    tuple(self.c.get('mask_chunk', [4, 12])), self.c['mask_p'], need)
        mx = self.c.get('max_frames', 160)
        if len(idx) > mx:
            idx = idx[np.linspace(0, len(idx) - 1, mx).round().astype(int)]
        video = read_clip([files[j] for j in idx], self.train, self.c.get('flip_p', 0.0))
        return dict(video=video, label=label, ref=toks, id=self.ids[i],
                    count_bin=count_bin(len(toks), self.bins))


def collate_continuous(batch):
    vlen = torch.tensor([b['video'].shape[0] for b in batch], dtype=torch.long)
    video = torch.zeros(len(batch), int(vlen.max()), *batch[0]['video'].shape[1:])
    for k, b in enumerate(batch):
        video[k, :b['video'].shape[0]] = b['video']
    labels = torch.tensor([t for b in batch for t in b['label']], dtype=torch.int32)
    label_len = torch.tensor([len(b['label']) for b in batch], dtype=torch.int32)
    return dict(video=video, vlen=vlen, labels=labels, label_len=label_len,
                count_bin=torch.tensor([b['count_bin'] for b in batch], dtype=torch.long),
                ref=[b['ref'] for b in batch], id=[b['id'] for b in batch])
