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


def update_config_from_cli(cfg, overrides=None):
    """Applies a list of 'key.subkey=value' overrides to a config dictionary."""
    for item in overrides or []:
        if '=' not in item:
            continue
        key, val = item.split('=', 1)
        node = cfg
        parts = key.split('.')
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def load_config(path, overrides=None):
    """Loads a YAML config file and optionally applies overrides."""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if overrides:
        update_config_from_cli(cfg, overrides)
    return cfg


# Alias for backwards compatibility
load_cfg = load_config


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    json.dump(obj, open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)


def load_pretrained_visual(visual, path, transfer_gam=True, verbose=True):
    """Load a 'visual' state dict saved by train_include.py into a VisualStage1 module.

    Only keys that exist in the target with identical shape are loaded; GAM keys can be excluded.
    Prints what was loaded and skipped so silent mismatches are visible."""
    if not os.path.exists(path):
        candidates = [
            'work_dir/include_visual.pt',
            'work_dir/include_pretrain/include_visual.pt',
            os.path.basename(path),
            os.path.join('work_dir', os.path.basename(path))
        ]
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
    ck = torch.load(path, map_location='cpu')
    sd = ck['visual'] if 'visual' in ck else ck
    own = visual.state_dict()
    loaded, skipped = {}, []
    for k, v in sd.items():
        if not transfer_gam and (k.startswith('gams.') or k.startswith('gcmm.')):
            skipped.append((k, 'gam/gcmm excluded (learning continuous context)'))
        elif k in own and own[k].shape == v.shape:
            loaded[k] = v
        elif transfer_gam:
            # Smart adapter for legacy GAM checkpoints (e.g. from older include_visual.pt)
            mapped = False
            if '.cluster.w' in k:
                k_tw, k_sw = k.replace('.cluster.w', '.cluster.t_w'), k.replace('.cluster.w', '.cluster.s_w')
                if k_tw in own and own[k_tw].shape == v.shape: loaded[k_tw] = v; mapped = True
                if k_sw in own and own[k_sw].shape == v.shape: loaded[k_sw] = v; mapped = True
            elif '.cluster.branches.' in k:
                if k.endswith('.0.weight') and v.dim() == 5:
                    k_t = k.replace('.cluster.branches.', '.cluster.t_branches.')
                    if k_t in own and own[k_t].shape == (v.shape[0], 1, v.shape[2]):
                        loaded[k_t] = torch.stack([v[i, i, :, 0, 0] for i in range(v.shape[0])]).unsqueeze(1)
                        mapped = True
                elif '.1.' in k:
                    k_t = k.replace('.cluster.branches.', '.cluster.t_branches.')
                    if k_t in own and own[k_t].shape == v.shape:
                        loaded[k_t] = v
                        mapped = True
            elif '.out.0.weight' in k:
                k_cb = k.replace('.out.0.weight', '.conv_block.0.weight')
                if k_cb in own and own[k_cb].shape == v.shape:
                    loaded[k_cb] = v
                    mapped = True
            elif '.out.1.' in k:
                k_cb = k.replace('.out.1.', '.conv_block.1.')
                if k_cb in own and own[k_cb].shape == v.shape:
                    loaded[k_cb] = v
                    mapped = True

            if not mapped:
                skipped.append((k, 'missing or shape mismatch'))
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
