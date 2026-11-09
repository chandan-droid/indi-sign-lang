"""Pure numpy/python: WER with sub/ins/del breakdown and CTC decoders (no torch import)."""
import numpy as np
from collections import defaultdict


def align_counts(ref, hyp):
    """Return (substitutions, deletions, insertions) from a minimum edit-distance alignment."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1])
            d[i, j] = min(sub, d[i - 1, j] + 1, d[i, j - 1] + 1)
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            S += int(ref[i - 1] != hyp[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return S, D, I


def wer_report(refs, hyps):
    """refs/hyps: lists of token lists. WER = (S + D + I) / total reference tokens, in percent."""
    S = D = I = N = 0
    for r, h in zip(refs, hyps):
        s, d, i = align_counts(r, h)
        S, D, I, N = S + s, D + d, I + i, N + len(r)
    N = max(N, 1)
    return dict(wer=100.0 * (S + D + I) / N, sub=S, dele=D, ins=I, n_ref=N,
                sub_rate=100.0 * S / N, del_rate=100.0 * D / N, ins_rate=100.0 * I / N)


def greedy_decode(scores, length, blank=0):
    """scores: [T, V] array (logits or probs). Collapse repeats, drop blanks."""
    ids = np.asarray(scores)[:length].argmax(-1)
    out, prev = [], -1
    for t in ids:
        if t != prev and t != blank:
            out.append(int(t))
        prev = t
    return out


def prefix_beam_search(probs, length, beam=10, blank=0, topk=20):
    """CTC prefix beam search on a [T, V] probability array (no language model)."""
    beams = {(): (1.0, 0.0)}  # prefix -> (p_blank_ending, p_nonblank_ending)
    for t in range(length):
        p = probs[t]
        cand = set(np.argsort(-p)[:topk].tolist())
        cand.add(blank)
        nxt = defaultdict(lambda: [0.0, 0.0])
        for prefix, (pb, pnb) in beams.items():
            tot = pb + pnb
            for c in cand:
                pc = float(p[c])
                if c == blank:
                    nxt[prefix][0] += tot * pc
                elif prefix and prefix[-1] == c:
                    nxt[prefix + (c,)][1] += pb * pc      # repeat only after a blank
                    nxt[prefix][1] += pnb * pc            # collapse into same label
                else:
                    nxt[prefix + (c,)][1] += tot * pc
        items = sorted(nxt.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:beam]
        z = sum(v[0] + v[1] for _, v in items) or 1.0     # renormalise to avoid underflow
        beams = {k: (v[0] / z, v[1] / z) for k, v in items}
    best = max(beams.items(), key=lambda kv: kv[1][0] + kv[1][1])[0]
    return list(best)
