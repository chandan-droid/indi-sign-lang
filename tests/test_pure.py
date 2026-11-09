"""Numpy-only tests: python tests/test_pure.py"""
import sys, random
sys.path.insert(0, '.')
import numpy as np
from ganet.metrics import align_counts, wer_report, greedy_decode, prefix_beam_search
from ganet.data.augment import temporal_mask, temporal_rescale

# WER breakdown
assert align_counts(list('abc'), list('abc')) == (0, 0, 0)
assert align_counts(list('abc'), list('abd')) == (1, 0, 0)
assert align_counts(list('abc'), list('ab')) == (0, 1, 0)
assert align_counts(list('ab'), list('abc')) == (0, 0, 1)
r = wer_report([list('abcd')], [list('axd')])
assert r['sub'] + r['dele'] + r['ins'] == 2 and abs(r['wer'] - 50.0) < 1e-6

# decoders: path  blank a a blank b b b blank a  -> [a, b, a]
V, path = 3, [0, 1, 1, 0, 2, 2, 2, 0, 1]
probs = np.full((len(path), V), 0.02)
for t, c in enumerate(path):
    probs[t, c] = 0.96
probs /= probs.sum(1, keepdims=True)
assert greedy_decode(probs, len(path)) == [1, 2, 1]
assert prefix_beam_search(probs, len(path), beam=5) == [1, 2, 1]
# repeated label separated by blank must survive: 1 0 1
p2 = np.array([[0.02, 0.96, 0.02], [0.96, 0.02, 0.02], [0.02, 0.96, 0.02]])
assert prefix_beam_search(p2, 3, beam=5) == [1, 1] and greedy_decode(p2, 3) == [1, 1]

# augmentations
random.seed(0)
idx = np.arange(100)
for _ in range(200):
    m = temporal_mask(idx, max_drop=0.25, chunk=(4, 12), p=1.0, min_keep=70)
    assert len(m) >= 70 and (np.diff(m) >= 1).all()
    s = temporal_rescale(idx, 0.8, 1.2)
    assert 80 <= len(s) <= 120 and s.min() >= 0 and s.max() <= 99
print('pure tests passed')
