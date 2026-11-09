"""Evaluate a saved checkpoint: python evaluate.py --ckpt work_dir/x/best.pt --split test --beam 10"""
import argparse, json
import torch
from torch.utils.data import DataLoader

from ganet.data.isl_csltr import read_manifest, ContinuousSignDataset, collate_continuous, Vocab
from ganet.models.ganet import build_model
from ganet.engine import evaluate
from ganet.utils import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--beam', type=int, default=10)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(a.ckpt, map_location='cpu')
    cfg, vocab = ck['cfg'], Vocab(ck['vocab'])
    d, m = cfg['data'], cfg['model']
    rows, splits = read_manifest(d['manifest']), json.load(open(d['splits']))
    ds = ContinuousSignDataset(rows, splits[a.split], vocab, False, d, m.get('num_pool', 2), m.get('glp_bins', 12))
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_continuous, num_workers=2)
    model = build_model(m, len(vocab)).to(device)
    model.load_state_dict(ck['model'])
    rep = evaluate(model, loader, vocab, device, beam=a.beam)
    print({k: round(v, 2) for k, v in rep.items() if isinstance(v, float)}, 'sub/del/ins =',
          rep['sub'], rep['dele'], rep['ins'])
    if a.out:
        save_json(rep, a.out)


if __name__ == '__main__':
    main()
