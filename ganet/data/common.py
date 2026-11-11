import glob, os, random
import cv2
import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_path(p):
    """Normalize Windows/Linux paths to work cross-platform."""
    if not p:
        return p
    p_norm = p.replace('\\', '/')
    if os.path.exists(p_norm):
        return p_norm
    # If absolute path from another machine (e.g. C:/.../frames/...), resolve to local frames/
    idx = p_norm.find('frames/')
    if idx != -1 and os.path.exists(p_norm[idx:]):
        return p_norm[idx:]
    return p_norm


def list_frames(d):
    d = normalize_path(d)
    files = sorted(glob.glob(os.path.join(d, '*.jpg')) + glob.glob(os.path.join(d, '*.png')))
    if not files:
        raise FileNotFoundError(f'no frames in {d}')
    return files


def read_clip(paths, train, flip_p=0.0, resize=256, size=224):
    """Shared by INCLUDE and ISL-CSLTR so both see identical preprocessing. Returns [T, 3, size, size]."""
    imgs = []
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            raise FileNotFoundError(p)
        if im.shape[0] != resize or im.shape[1] != resize:
            im = cv2.resize(im, (resize, resize))
        imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    arr = np.stack(imgs)
    if train:
        y, x = random.randint(0, resize - size), random.randint(0, resize - size)
        flip = random.random() < flip_p
    else:
        y = x = (resize - size) // 2
        flip = False
    arr = arr[:, y:y + size, x:x + size]
    if flip:
        arr = arr[:, :, ::-1]
    arr = (arr.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2)
