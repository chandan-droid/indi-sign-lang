"""Train GANet on a continuous-sign dataset (ISL-CSLTR). Supports baseline/ablation flags via config."""
import argparse, os, time, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ganet.utils import set_seed, load_cfg, save_json, load_pretrained_visual
from ganet.data.isl_csltr import (read_manifest, build_vocab, ContinuousSignDataset, collate_continuous, Vocab)
from ganet.models.ganet import build_model
from ganet.losses import total_loss
from ganet.engine import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', nargs='*', default=[], help='overrides, e.g. seed=1 train.epochs=2')
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.set)
    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d, t, m, w = cfg['data'], cfg['train'], cfg['model'], cfg['loss']
    wd = cfg['work_dir']
    os.makedirs(wd, exist_ok=True)

    rows = read_manifest(d['manifest'])
    splits_path = d['splits']
    if not os.path.exists(splits_path):
        alt = 'data/isl_splits.json' if splits_path == 'splits/isl_signer_v1.json' else 'splits/isl_signer_v1.json'
        if os.path.exists(alt):
            splits_path = alt
    splits = json.load(open(splits_path))
    vocab = build_vocab(rows, splits['train'], d.get('lowercase', True))
    vocab.save(os.path.join(wd, 'vocab.json'))
    print('vocab (incl. blank):', len(vocab), {k: len(v) for k, v in splits.items()})

    pool = m.get('num_pool', 2)
    mk = lambda s, train: ContinuousSignDataset(rows, splits[s], vocab, train, d, pool, m.get('glp_bins', 12))
    nw = t.get('num_workers', 4)
    ld = dict(train=DataLoader(mk('train', True), batch_size=t['batch_size'], shuffle=True, drop_last=True,
                               num_workers=nw, collate_fn=collate_continuous, pin_memory=True),
              dev=DataLoader(mk('dev', False), batch_size=1, num_workers=nw, collate_fn=collate_continuous),
              test=DataLoader(mk('test', False), batch_size=1, num_workers=nw, collate_fn=collate_continuous))

    model = build_model(m, len(vocab)).to(device)
    pre = cfg.get('pretrained')
    if pre and pre.get('path'):
        load_pretrained_visual(model.visual, pre['path'], pre.get('transfer_gam', True))

    visual_params = list(model.visual.parameters())
    other = [p for n_, p in model.named_parameters() if not n_.startswith('visual.')]
    opt = torch.optim.Adam([dict(params=visual_params, lr=t['lr_visual']), dict(params=other, lr=t['lr'])],
                           weight_decay=t['weight_decay'])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, t['milestones'], t.get('gamma', 0.2))
    amp = t.get('amp', True) and device.type == 'cuda'
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    warm = (pre or {}).get('freeze_epochs', 0) if pre and pre.get('path') else 0
    best, best_path, log = 1e9, os.path.join(wd, 'best.pt'), []

    for ep in range(t['epochs']):
        for p in visual_params:
            p.requires_grad = ep >= warm                  # warm up new heads while the visual stage is frozen
        model.train()
        if ep < warm:
            model.visual.eval()                           # keep BatchNorm statistics fixed too
        t0, agg, nb = time.time(), {}, 0
        for batch in ld['train']:
            video, vlen = batch['video'].to(device), batch['vlen'].to(device)
            try:
                autocast_ctx = torch.amp.autocast('cuda', dtype=torch.float16, enabled=amp)
            except (AttributeError, TypeError):
                autocast_ctx = torch.cuda.amp.autocast(enabled=amp)
            with autocast_ctx:
                out = model(video, vlen)
            loss, parts = total_loss(out, batch, w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
        sched.step()
        rep = evaluate(model, ld['dev'], vocab, device, beam=1, amp=amp)
        row = dict(epoch=ep, dev_wer=rep['wer'], dev_sub=rep['sub'], dev_del=rep['dele'], dev_ins=rep['ins'],
                   **{f'loss_{k}': v / max(nb, 1) for k, v in agg.items()})
        log.append(row)
        print(json.dumps(row), f'({time.time() - t0:.0f}s)', flush=True)
        if rep['wer'] < best:                              # model selection on DEV only
            best = rep['wer']
            torch.save(dict(model=model.state_dict(), vocab=vocab.itos[1:], cfg=cfg, epoch=ep, dev_wer=best), best_path)

    ck = torch.load(best_path, map_location='cpu')
    model.load_state_dict(ck['model'])
    dev = evaluate(model, ld['dev'], vocab, device, beam=t.get('beam', 10), amp=amp)
    test = evaluate(model, ld['test'], vocab, device, beam=t.get('beam', 10), amp=amp)   # test touched once
    save_json(dict(seed=cfg['seed'], best_epoch=ck['epoch'], dev={k: v for k, v in dev.items() if k != 'predictions'},
                   test={k: v for k, v in test.items() if k != 'predictions'}, log=log, cfg=cfg),
              os.path.join(wd, 'results.json'))
    save_json(test['predictions'], os.path.join(wd, 'test_predictions.json'))
    print('DEV ', {k: round(v, 2) for k, v in dev.items() if k not in ('predictions',) and isinstance(v, float)})
    print('TEST', {k: round(v, 2) for k, v in test.items() if k not in ('predictions',) and isinstance(v, float)})


if __name__ == '__main__':
    main()
