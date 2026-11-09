"""Pre-train Stage 1 (SMM + GCMM + GAM) on INCLUDE isolated signs; saves ONLY the visual stage weights."""
import argparse, os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ganet.utils import set_seed, load_cfg, save_json
from ganet.data.include import read_include_manifest, IncludeClips
from ganet.models.include_model import IncludeClassifier


def accuracy(model, loader, device, amp):
    model.eval()
    correct = n = 0
    with torch.no_grad():
        for x, y in loader:
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp and device.type == 'cuda'):
                p = model(x.to(device)).argmax(-1).cpu()
            correct += (p == y).sum().item()
            n += len(y)
    return correct / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', nargs='*', default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.set)
    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d, t, m = cfg['data'], cfg['train'], cfg['model']
    os.makedirs(cfg['work_dir'], exist_ok=True)

    rows, c2i = read_include_manifest(d['manifest'])
    mk = lambda split, train: IncludeClips([r for r in rows if r['split'] == split], c2i, train,
                                           d.get('num_frames', 32), d.get('flip_p', 0.0))
    ds = {s: mk(s, s == 'train') for s in ('train', 'val', 'test')}
    ld = {s: DataLoader(ds[s], batch_size=t['batch_size'], shuffle=(s == 'train'), drop_last=(s == 'train'),
                        num_workers=t.get('num_workers', 4), pin_memory=True) for s in ds if len(ds[s])}
    print({s: len(ds[s]) for s in ds}, 'classes:', len(c2i))

    model = IncludeClassifier(len(c2i), m.get('gcmm', True), m.get('gam', True),
                              m.get('imagenet', True), m.get('temporal_mix', True)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t['epochs'])
    crit = nn.CrossEntropyLoss(label_smoothing=t.get('label_smoothing', 0.1))
    amp = t.get('amp', True) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best, ckpt = -1.0, os.path.join(cfg['work_dir'], 'include_visual.pt')

    for ep in range(t['epochs']):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y in ld['train']:
            x, y = x.to(device), y.to(device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                loss = crit(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            tot, nb = tot + loss.item(), nb + 1
        sched.step()
        acc = accuracy(model, ld['val'], device, amp)
        print(f'ep {ep:3d} loss {tot / max(nb, 1):.4f} val_acc {acc:.4f} ({time.time() - t0:.0f}s)', flush=True)
        if acc > best:
            best = acc
            torch.save(dict(visual=model.visual.state_dict(), model=model.state_dict(), val_acc=acc,
                            epoch=ep, classes=list(c2i), model_cfg=m), ckpt)

    res = dict(best_val_acc=best, checkpoint=ckpt)
    if 'test' in ld:
        model.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])   # best epoch, full model
        res['test_acc'] = accuracy(model, ld['test'], device, amp)
    save_json(res, os.path.join(cfg['work_dir'], 'results.json'))
    print(res)


if __name__ == '__main__':
    main()
