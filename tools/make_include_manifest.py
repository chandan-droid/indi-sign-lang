"""Build the INCLUDE manifest (frame_dir,label,split) from extracted frames laid out as FRAMES/<...>/<label>/<clip>/*.jpg
Official split: pass --train_list/--test_list (text files, one clip folder name per line, taken from train.csv/test.csv).
Without them a stratified random split is made (NOT the official one). A validation set is carved from train."""
import argparse, csv, glob, os, random, re
from collections import defaultdict


def clean(name):
    return re.sub(r'^[\d\s\.\-_]+', '', name).strip().lower()      # '12. loud' -> 'loud'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--train_list', default=None)
    ap.add_argument('--test_list', default=None)
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--test_frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=3407)
    a = ap.parse_args()
    clips = sorted({os.path.dirname(f) for f in glob.glob(os.path.join(a.frames, '**', '*.jpg'), recursive=True)})
    items = []
    for c in clips:
        parent = os.path.basename(os.path.dirname(c))
        if parent.lower() in ('extra', 'supplementary', 'additional'):
            parent = os.path.basename(os.path.dirname(os.path.dirname(c)))
        items.append((c, clean(parent)))
    random.seed(a.seed)
    train_names = set(open(a.train_list).read().split()) if a.train_list else None
    test_names = set(open(a.test_list).read().split()) if a.test_list else None
    per = defaultdict(list)
    for c, l in items:
        per[l].append(c)
    rows = []
    for l, cs in per.items():
        random.shuffle(cs)
        for k, c in enumerate(cs):
            stem = os.path.basename(c)
            if train_names is not None or test_names is not None:
                split = 'test' if stem in (test_names or set()) else 'train'
            else:
                split = 'test' if k < max(1, int(len(cs) * a.test_frac)) else 'train'
            rows.append(dict(frame_dir=c, label=l, split=split))
    by = defaultdict(list)
    for r in rows:
        if r['split'] == 'train':
            by[r['label']].append(r)
    for l, rs in by.items():                                        # validation carved from train
        for r in rs[:max(1, int(len(rs) * a.val_frac))]:
            r['split'] = 'val'
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['frame_dir', 'label', 'split'])
        w.writeheader()
        w.writerows(rows)
    cnt = defaultdict(int)
    for r in rows:
        cnt[r['split']] += 1
    print(dict(cnt), 'classes:', len(per))


if __name__ == '__main__':
    main()
