import glob, os, random
import cv2
import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_path(p):
    """Normalize Windows/Linux paths to work cross-platform. Handles both str and list/tuple."""
    if not p:
        return p
    if isinstance(p, (list, tuple)):
        return [normalize_path(x) for x in p]
    p_norm = str(p).replace('\\', '/')
    if os.path.exists(p_norm):
        return p_norm
    idx = p_norm.find('frames/')
    if idx != -1 and os.path.exists(p_norm[idx:]):
        return p_norm[idx:]
    return p_norm


def get_video_frame_count(vid_path):
    """Returns the total number of frames in a video file without loading all into memory."""
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        return 0
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(cnt, 1)


def list_frames(d):
    """
    Returns image paths if 'd' is a directory of images (.jpg/.png),
    or returns [d] if 'd' is a video file.
    """
    d = normalize_path(d)
    if isinstance(d, str) and os.path.isfile(d) and d.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return [d]

    files = sorted(glob.glob(os.path.join(d, '*.jpg')) + glob.glob(os.path.join(d, '*.png')))
    if not files:
        raise FileNotFoundError(f'no frames in {d}')
    return files


def read_clip_sampled(d, indices, train, flip_p=0.0, resize=256, size=224):
    """
    Loads specific sampled frame indices from either:
    1. A directory of image frames or list of frame paths
    2. An .mp4 / video file (seeks directly to indices)
    Returns [T, 3, size, size].
    """
    imgs = []

    # CASE A: Input is a single video file path (.mp4)
    if isinstance(d, str):
        d_norm = normalize_path(d)
        if os.path.isfile(d_norm) and d_norm.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            cap = cv2.VideoCapture(d_norm)
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open video: {d_norm}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            safe_indices = [min(max(0, int(i)), max(0, total_frames - 1)) for i in indices]

            current_idx = 0
            idx_set = set(safe_indices)
            frame_cache = {}

            max_idx = max(safe_indices) if safe_indices else 0
            while cap.isOpened() and current_idx <= max_idx:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if current_idx in idx_set:
                    if frame.shape[0] != resize or frame.shape[1] != resize:
                        frame = cv2.resize(frame, (resize, resize))
                    frame_cache[current_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                current_idx += 1
            cap.release()

            fallback = list(frame_cache.values())[-1] if frame_cache else np.zeros((resize, resize, 3), dtype=np.uint8)
            for i in safe_indices:
                imgs.append(frame_cache.get(i, fallback))
        else:
            raise ValueError(f"Expected video file or frame list, got: {d}")

    # CASE B: Input is a list or tuple of frame image paths
    elif isinstance(d, (list, tuple)):
        for p in d:
            p_norm = normalize_path(p)
            im = cv2.imread(p_norm)
            if im is None:
                raise FileNotFoundError(p_norm)
            if im.shape[0] != resize or im.shape[1] != resize:
                im = cv2.resize(im, (resize, resize))
            imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError(f"Unsupported input type for read_clip_sampled: {type(d)}")

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


def read_clip(paths, train, flip_p=0.0, resize=256, size=224):
    """Backwards compatible read_clip for list of file paths."""
    return read_clip_sampled(paths, list(range(len(paths))), train, flip_p, resize, size)
