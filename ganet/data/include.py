"""INCLUDE isolated-sign dataset. Manifest CSV columns: frame_dir,label,split (train/val/test)."""
import csv, random
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import list_frames, read_clip


def read_include_manifest(path):
    rows = list(csv.DictReader(open(path)))
    classes = sorted({r['label'] for r in rows if r['split'] == 'train'})
    return rows, {c: i for i, c in enumerate(classes)}


class IncludeClips(Dataset):
    def __init__(self, rows, class_to_id, train, num_frames=32, flip_p=0.0):
        self.rows = [r for r in rows if r['label'] in class_to_id]
        self.c2i, self.train, self.T, self.flip_p = class_to_id, train, num_frames, flip_p

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        files = list_frames(r['frame_dir'])
        n = len(files)
        edges = np.linspace(0, n, self.T + 1).astype(int)
        if self.train:     # one random frame per temporal segment (keeps order, adds jitter)
            idx = [random.randint(edges[k], max(edges[k], edges[k + 1] - 1)) for k in range(self.T)]
        else:
            idx = [min((edges[k] + edges[k + 1]) // 2, n - 1) for k in range(self.T)]
        return read_clip([files[j] for j in idx], self.train, self.flip_p), self.c2i[r['label']]
