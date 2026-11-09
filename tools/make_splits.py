"""Create frozen split JSONs from a manifest CSV (id,frame_dir,signer,text).

  signer   : python tools/make_splits.py --manifest M.csv --mode signer --dev S2 --test S7 --out splits/isl_signer_v1.json
  loso     : python tools/make_splits.py --manifest M.csv --mode loso --out_dir splits/   (one file per held-out signer)
  sentence : python tools/make_splits.py --manifest M.csv --mode sentence --dev_frac 0.1 --test_frac 0.15 --out ...
Prints leakage checks: unseen test words are words CTC can never learn."""
import argparse, csv, json, os, random, re


def toks(t):
    return re.sub(r'[^\w\s]', ' ', t.lower()).split()


def report(rows, sp, name):
    seen = {w for i in sp['train'] for w in toks(rows[i]['text'])}
    unseen = {w for i in sp['test'] for w in toks(rows[i]['text'])} - seen
    sig = {k: sorted({rows[i]['signer'] for i in v}) for k, v in sp.items()}
    print(name, {k: len(v) for k, v in sp.items()}, 'signers', sig, '| test words unseen in train:', len(unseen))
    assert not (set(sp['train']) & set(sp['test'])) and not (set(sp['train']) & set(sp['dev']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--mode', choices=['signer', 'loso', 'sentence'], required=True)
    ap.add_argument('--dev', default=None)
    ap.add_argument('--test', default=None)
    ap.add_argument('--dev_frac', type=float, default=0.1)
    ap.add_argument('--test_frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=3407)
    ap.add_argument('--out', default=None)
    ap.add_argument('--out_dir', default='splits')
    a = ap.parse_args()
    rows = {r['id']: r for r in csv.DictReader(open(a.manifest))}
    ids = sorted(rows)
    signers = sorted({r['signer'] for r in rows.values()})
    print('signers found:', len(signers), signers)

    def by_signer(dev, test):
        return dict(train=[i for i in ids if rows[i]['signer'] not in (dev, test)],
                    dev=[i for i in ids if rows[i]['signer'] == dev],
                    test=[i for i in ids if rows[i]['signer'] == test])

    if a.mode == 'signer':
        sp = by_signer(a.dev, a.test)
        report(rows, sp, 'signer-independent')
        json.dump(sp, open(a.out, 'w'), indent=1)
    elif a.mode == 'loso':
        os.makedirs(a.out_dir, exist_ok=True)
        for k, s in enumerate(signers):
            sp = by_signer(signers[(k + 1) % len(signers)], s)
            report(rows, sp, f'loso test={s}')
            json.dump(sp, open(os.path.join(a.out_dir, f'loso_{s}.json'), 'w'), indent=1)
    else:
        random.seed(a.seed)
        sents = sorted({' '.join(toks(r['text'])) for r in rows.values()})
        random.shuffle(sents)
        nt, nd = int(len(sents) * a.test_frac), int(len(sents) * a.dev_frac)
        test_s, dev_s = set(sents[:nt]), set(sents[nt:nt + nd])
        key = lambda i: ' '.join(toks(rows[i]['text']))
        sp = dict(train=[i for i in ids if key(i) not in test_s | dev_s],
                  dev=[i for i in ids if key(i) in dev_s], test=[i for i in ids if key(i) in test_s])
        report(rows, sp, 'sentence-held-out')
        json.dump(sp, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
