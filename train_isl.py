"""Training loop with NaN/Inf loss explosion guard, scaler stabilization, and batch safety."""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ganet.data import (
    ContinuousSignDataset,
    build_vocab,
    collate_continuous,
    read_manifest,
)
from ganet.evaluate import evaluate
from ganet.losses import total_loss
from ganet.models.ganet import build_model
from ganet.utils import (
    load_config,
    load_pretrained_visual,
    set_seed,
    update_config_from_cli,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--set', nargs='*', default=[], help='key=val overrides')
    args = parser.parse_args()

    cfg = load_config(args.config)
    update_config_from_cli(cfg, args.set)
    set_seed(cfg.get('seed', 3407))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d, m, t, w = cfg['data'], cfg['model'], cfg['train'], cfg.get('loss', {})
    wd = cfg.get('work_dir', 'work_dir/run')
    os.makedirs(wd, exist_ok=True)

    rows = read_manifest(d['manifest'])
    splits_path = d['splits']
    if not os.path.exists(splits_path):
        alt = os.path.join(os.path.dirname(d['manifest']), os.path.basename(splits_path))
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
    other_params = [p for n, p in model.named_parameters() if not n.startswith('visual.')]
    opt = torch.optim.Adam([
        dict(params=other_params, lr=t['lr'], weight_decay=t.get('weight_decay', 1e-4)),
        dict(params=visual_params, lr=t.get('lr_visual', t['lr']), weight_decay=t.get('weight_decay', 1e-4)),
    ])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=t.get('milestones', [20, 30]),
                                                 gamma=t.get('gamma', 0.5))
    amp = t.get('amp', True) and device == 'cuda'
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    warm = (pre or {}).get('freeze_epochs', 0) if pre and pre.get('path') else 0
    best, best_path, log = 1e9, os.path.join(wd, 'best.pt'), []

    for ep in range(t['epochs']):
        for p in visual_params:
            p.requires_grad = ep >= warm
        model.train()
        model.visual.eval()
        t0, agg, nb = time.time(), {}, 0

        for batch_idx, batch in enumerate(ld['train']):
            video, vlen = batch['video'].to(device), batch['vlen'].to(device)
            try:
                autocast_ctx = torch.amp.autocast('cuda', dtype=torch.float16, enabled=amp)
            except (AttributeError, TypeError):
                autocast_ctx = torch.cuda.amp.autocast(enabled=amp)

            with autocast_ctx:
                out = model(video, vlen)

            loss, parts = total_loss(out, batch, w)

            # --- CRITICAL SAFEGUARD: Skip any batch with NaN or Inf loss ---
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[!] Warning: Epoch {ep} batch {batch_idx} produced non-finite loss ({float(loss)}). Skipping step.", flush=True)
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            # Clip gradient norm to prevent gradient explosion
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"[!] Warning: Epoch {ep} batch {batch_idx} produced Inf/NaN gradients. Skipping step.", flush=True)
                opt.zero_grad(set_to_none=True)
                scaler.update()
                continue

            scaler.step(opt)
            scaler.update()

            for k, v in parts.items():
                if not (torch.isnan(torch.tensor(v)) or torch.isinf(torch.tensor(v))):
                    agg[k] = agg.get(k, 0.0) + v
            nb += 1

            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                c_loss = parts.get('ctc', 0.0)
                print(f"[ep {ep:02d}] batch {batch_idx+1:04d}/{len(ld['train'])} (loss_ctc={c_loss:.4f}, {elapsed:.0f}s)", flush=True)

        sched.step()
        rep = evaluate(model, ld['dev'], vocab, device, beam=1, amp=amp)
        row = dict(epoch=ep, dev_wer=rep['wer'], dev_sub=rep['sub'], dev_del=rep['dele'], dev_ins=rep['ins'],
                   **{f'loss_{k}': v / max(nb, 1) for k, v in agg.items()})
        log.append(row)
        print(json.dumps(row), f'({time.time() - t0:.0f}s)', flush=True)

        if rep['wer'] < best:
            best = rep['wer']
            torch.save(dict(model=model.state_dict(), vocab=vocab.itos[1:], cfg=cfg, epoch=ep, dev_wer=best), best_path)

    # Final evaluation on DEV and TEST using best checkpoint
    if os.path.exists(best_path):
        ck = torch.load(best_path, map_location='cpu')
        model.load_state_dict(ck['model'])
    d_rep = evaluate(model, ld['dev'], vocab, device, beam=t.get('beam', 10), amp=amp)
    t_rep = evaluate(model, ld['test'], vocab, device, beam=t.get('beam', 10), amp=amp)
    print('DEV ', d_rep)
    print('TEST', t_rep)
    json.dump(dict(dev=d_rep, test=t_rep, log=log), open(os.path.join(wd, 'results.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
