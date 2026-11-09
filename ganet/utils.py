import json, os, random
import numpy as np
import torch
import yaml


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cfg(path, overrides=None):
    cfg = yaml.safe_load(open(path))
    for item in overrides or []:                                   # e.g. --set train.epochs=2
        key, val = item.split('=', 1)
        node = cfg
        parts = key.split('.')
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    json.dump(obj, open(path, 'w'), indent=1, ensure_ascii=False)


def load_pretrained_visual(visual, path, transfer_gam=True, verbose=True):
    """Load a 'visual' state dict saved by train_include.py into a VisualStage1 module.

    Only keys that exist in the target with identical shape are loaded; GAM keys can be excluded.
    Prints what was loaded and skipped so silent mismatches are visible."""
    ck = torch.load(path, map_location='cpu')
    sd = ck['visual'] if 'visual' in ck else ck
    own = visual.state_dict()
    loaded, skipped = {}, []
    for k, v in sd.items():
        if not transfer_gam and k.startswith('gams.'):
            skipped.append((k, 'gam excluded'))
        elif k in own and own[k].shape == v.shape:
            loaded[k] = v
        else:
            skipped.append((k, 'missing or shape mismatch'))
    visual.load_state_dict(loaded, strict=False)
    missing = [k for k in own if k not in loaded]
    if verbose:
        print(f'[pretrained] loaded {len(loaded)}/{len(own)} visual tensors from {path}; '
              f'skipped {len(skipped)}; still random-init {len(missing)}')
        for k, why in skipped[:5]:
            print('   skipped:', k, why)
    assert len(loaded) > 0, 'no tensors transferred: key names do not match'
    return ck
