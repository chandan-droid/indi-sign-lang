"""Continuous-sign dataset (supports ISL-CSLTR image dirs and PHOENIX .mp4 videos).

Manifest CSV columns: id,frame_dir,signer,text
Split JSON: {"train": [ids], "dev": [ids], "test": [ids]}
"""
import csv, json, re, os
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import list_frames, read_clip_sampled, get_video_frame_count, normalize_path
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
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [fn.strip() for fn in (reader.fieldnames or [])]
        rows = {}
        for r in reader:
            row_clean = {k.strip(): v.strip() for k, v in r.items() if k is not None}
            row_id = row_clean.get('id') or list(row_clean.values())[0]
            rows[row_id] = row_clean
        return rows


def build_vocab(rows, train_ids, lowercase=True):
    words = sorted({w for i in train_ids for w in tokenize(rows[i]['text'], lowercase)})
    return Vocab(words)


def count_bin(n_words, num_bins):
    return min(max(int(n_words) - 1, 0), num_bins - 1)


class ContinuousSignDataset(Dataset):
    """
    Dataset class compatible with:
    ContinuousSignDataset(rows, ids, vocab, train, d, pool, bins) [called by train_isl.py]
    and
    ContinuousSignDataset(rows, ids, vocab, cfg, train=True)
    """
    def __init__(self, manifest_rows, ids, vocab, train=True, cfg=None, num_pool=0, glp_bins=12):
        self.rows = manifest_rows
        self.ids = ids
        self.vocab = vocab
        # Flexible signature: train can be bool or dict
        if isinstance(train, dict) and cfg is None:
            self.c = train
            self.train = True
            self.pool = 2 ** self.c.get('num_pool', 0)
            self.bins = self.c.get('glp_bins', 12)
        else:
            self.train = train
            self.c = cfg or {}
            self.pool = 2 ** num_pool
            self.bins = glp_bins

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        r = self.rows[self.ids[i]]
        toks = tokenize(r['text'], self.c.get('lowercase', True))
        label = self.vocab.encode(toks)
        target_path = normalize_path(r['frame_dir'])

        # Check if single video file (.mp4) or directory of frames
        is_video = os.path.isfile(target_path) and target_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

        if is_video:
            num_frames = get_video_frame_count(target_path)
            if num_frames <= 0:
                num_frames = 30  # fallback
            files = None
        else:
            files = list_frames(target_path)
            num_frames = len(files)

        idx = np.arange(num_frames)
        min_needed = (len(label) + 1) * self.pool
        if len(idx) < min_needed and num_frames > 0:
            idx = np.round(np.linspace(0, num_frames - 1, min_needed)).astype(int)

        if self.train:
            if self.c.get('temporal_scale', True):
                idx = temporal_rescale(idx, *self.c.get('scale_range', [0.8, 1.2]))
            if self.c.get('mask_p', 0.0) > 0:
                repeats = sum(a == b for a, b in zip(label, label[1:]))
                need = (len(label) + repeats + 1) * self.pool
                idx = temporal_mask(idx, self.c.get('mask_max_drop', 0.25),
                                    tuple(self.c.get('mask_chunk', [4, 12])), self.c['mask_p'], need)
            if len(idx) < min_needed and num_frames > 0:
                idx = np.round(np.linspace(0, num_frames - 1, min_needed)).astype(int)

        mx = self.c.get('max_frames', 160)
        if len(idx) > mx:
            idx = idx[np.linspace(0, len(idx) - 1, mx).round().astype(int)]

        # Sample and read frames
        if is_video:
            video = read_clip_sampled(target_path, idx, self.train, self.c.get('flip_p', 0.0))
        else:
            sampled_files = [files[j] for j in idx]
            video = read_clip_sampled(sampled_files, list(range(len(sampled_files))), self.train, self.c.get('flip_p', 0.0))

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
