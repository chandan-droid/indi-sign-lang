"""Index-level augmentations (numpy only). They operate on frame indices before any image is read."""
import random
import numpy as np


def temporal_rescale(idx, lo=0.8, hi=1.2):
    """Randomly stretch/compress the sequence length by a factor in [lo, hi] (paper: +/-20%)."""
    n = len(idx)
    new_n = max(1, int(round(n * random.uniform(lo, hi))))
    pos = np.linspace(0, n - 1, new_n).round().astype(int)
    return np.asarray(idx)[pos]


def temporal_mask(idx, max_drop=0.25, chunk=(4, 12), p=0.5, min_keep=0):
    """Drop random consecutive chunks of frames (training only).

    max_drop: maximum fraction of frames removed. chunk: (min, max) chunk length.
    min_keep: if fewer frames would remain, return the input unchanged (keeps CTC feasible).
    """
    idx = np.asarray(idx)
    n = len(idx)
    if n < 2 or random.random() > p:
        return idx
    keep = np.ones(n, dtype=bool)
    budget = int(n * max_drop)
    while budget > 0:
        length = min(random.randint(chunk[0], chunk[1]), budget)
        start = random.randint(0, max(0, n - length))
        keep[start:start + length] = False
        budget -= length
    out = idx[keep]
    return out if len(out) >= max(min_keep, 1) else idx
