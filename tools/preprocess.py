"""Master data preprocessing orchestrator for GANet ISL.

Handles both INCLUDE (isolated pre-training) and ISL-CSLTR (continuous recognition):
  1. Multi-threaded / parallel frame extraction (resized to 256x256 JPG).
  2. Manifest generation for INCLUDE (frame_dir, label, split).
  3. Manifest generation for ISL-CSLTR (id, frame_dir, signer, text).
  4. Train / Dev / Test split generation with leakage & CTC feasibility checks.

Usage examples:
  python tools/preprocess.py --dataset include
  python tools/preprocess.py --dataset isl
  python tools/preprocess.py --all
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


def clean_name(name):
    """Normalize label name: '12. Loud' -> 'loud'."""
    return re.sub(r'^[\d\s\.\-_]+', '', name).strip().lower()


def extract_single_video(video_path, out_dir, size=256, every=1):
    """Extract frames for a single video. Returns (video_path, frame_count, error)."""
    import cv2
    try:
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            # Already extracted
            return video_path, len(glob.glob(os.path.join(out_dir, '*.jpg'))), None
        
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return video_path, 0, f"Cannot open video: {video_path}"
        
        n, kept = 0, 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if n % every == 0:
                h, w = im.shape[:2]
                if h != size or w != size:
                    im = cv2.resize(im, (size, size))
                out_path = os.path.join(out_dir, f"{kept:06d}.jpg")
                cv2.imwrite(out_path, im, [cv2.IMWRITE_JPEG_QUALITY, 92])
                kept += 1
            n += 1
        cap.release()
        if kept == 0:
            return video_path, 0, "No frames read from video"
        return video_path, kept, None
    except Exception as e:
        return video_path, 0, str(e)


def extract_frames_parallel(video_files, video_root, out_root, size=256, every=1, workers=4):
    """Extract all video_files in parallel."""
    tasks = []
    print(f"[*] Extracting frames for {len(video_files)} videos with {workers} workers...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for f in video_files:
            rel = os.path.splitext(os.path.relpath(f, video_root))[0]
            out = os.path.join(out_root, rel)
            tasks.append(executor.submit(extract_single_video, f, out, size, every))
        
        done = 0
        total = len(tasks)
        for fut in as_completed(tasks):
            _, count, err = fut.result()
            if err:
                print(f"    [!] Error: {err}")
            done += 1
            if done % 100 == 0 or done == total:
                print(f"    Processed {done}/{total} videos ({done*100.0/total:.1f}%)", flush=True)


def process_include(raw_dir="raw_data/INCLUDE", frames_dir="frames/include",
                    manifest_out="data/include_manifest.csv",
                    val_frac=0.1, test_frac=0.2, seed=3407, workers=4, size=256):
    """Preprocess INCLUDE dataset: extract frames and build manifest."""
    print("=" * 60)
    print("Pre-processing INCLUDE (Isolated Sign Language)")
    print("=" * 60)
    if not os.path.isdir(raw_dir):
        print(f"[!] Raw directory not found: {raw_dir}")
        print("    Please place the INCLUDE dataset into raw_data/INCLUDE/")
        return False

    extensions = ['mp4', 'MOV', 'mov', 'avi', 'mkv']
    videos = [f for ext in extensions for f in glob.glob(os.path.join(raw_dir, '**', f'*.{ext}'), recursive=True)]
    if not videos:
        print(f"[!] No video files found in {raw_dir}")
        return False
    print(f"[*] Found {len(videos)} video files in {raw_dir}")

    # 1. Extract frames
    extract_frames_parallel(videos, raw_dir, frames_dir, size=size, workers=workers)

    # 2. Build manifest
    clips = sorted({os.path.dirname(f) for f in glob.glob(os.path.join(frames_dir, '**', '*.jpg'), recursive=True)})
    if not clips:
        print(f"[!] No frames found in {frames_dir}")
        return False

    # Extract label name: handles Category/SignName/clip or SignName/clip
    items = []
    for c in clips:
        c_norm = c.replace('\\', '/')
        parent = os.path.basename(os.path.dirname(c_norm))
        if parent.lower() in ('extra', 'supplementary', 'additional'):
            parent = os.path.basename(os.path.dirname(os.path.dirname(c_norm)))
        label = clean_name(parent)
        items.append((c_norm, label))

    import random
    random.seed(seed)
    per_class = defaultdict(list)
    for c, l in items:
        per_class[l].append(c)

    rows = []
    for l, cs in per_class.items():
        random.shuffle(cs)
        for k, c in enumerate(cs):
            split = 'test' if k < max(1, int(len(cs) * test_frac)) else 'train'
            rows.append(dict(frame_dir=c, label=l, split=split))

    # Carve validation from train
    by_label = defaultdict(list)
    for r in rows:
        if r['split'] == 'train':
            by_label[r['label']].append(r)
    for l, rs in by_label.items():
        for r in rs[:max(1, int(len(rs) * val_frac))]:
            r['split'] = 'val'

    os.makedirs(os.path.dirname(manifest_out) or '.', exist_ok=True)
    with open(manifest_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['frame_dir', 'label', 'split'])
        w.writeheader()
        w.writerows(rows)

    counts = defaultdict(int)
    for r in rows:
        counts[r['split']] += 1
    print(f"[✓] INCLUDE manifest saved to {manifest_out}")
    print(f"    Total clips: {len(rows)} across {len(per_class)} classes.")
    print(f"    Split distribution: {dict(counts)}")
    return True


def auto_detect_isl_columns(csv_path):
    """Auto-detect column names from ISL labels CSV."""
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, [])
    header_lower = [h.strip().lower() for h in header]
    
    # ID column candidates
    id_candidates = ['video', 'video_id', 'id', 'filename', 'file', 'clip', 'clip_id']
    id_col = header[0]
    for c in id_candidates:
        if c in header_lower:
            id_col = header[header_lower.index(c)]
            break

    # Text / sentence column candidates
    text_candidates = ['sentence', 'text', 'gloss', 'annotation', 'transcription', 'label', 'words']
    text_col = header[1] if len(header) > 1 else header[0]
    for c in text_candidates:
        if c in header_lower:
            text_col = header[header_lower.index(c)]
            break

    # Signer column candidates
    signer_candidates = ['signer', 'signer_id', 'subject', 'speaker', 'user']
    signer_col = None
    for c in signer_candidates:
        if c in header_lower:
            signer_col = header[header_lower.index(c)]
            break

    return id_col, text_col, signer_col


def process_isl(raw_dir="raw_data/ISL-CSLTR",
                videos_dir=None, labels_file=None,
                frames_dir="frames/isl",
                manifest_out="data/isl_manifest.csv",
                splits_out="splits/isl_signer_v1.json",
                dev_signer=None, test_signer=None,
                workers=4, size=256):
    """Preprocess ISL-CSLTR dataset: extract frames, build manifest, and generate splits."""
    print("=" * 60)
    print("Pre-processing ISL-CSLTR (Continuous Sign Language Recognition)")
    print("=" * 60)
    
    videos_dir = videos_dir or os.path.join(raw_dir, "videos")
    labels_file = labels_file or os.path.join(raw_dir, "labels.csv")

    # Check if pre-extracted Frames_Sentence_Level directory exists (official Kaggle dataset structure)
    from tools.make_isl_cslrt_manifest import find_sentence_frames_dir, generate_manifest_and_splits
    sentence_frames = find_sentence_frames_dir(raw_dir)
    if sentence_frames:
        print(f"[*] Found pre-extracted Frames_Sentence_Level at: {sentence_frames}")
        generate_manifest_and_splits(
            sentence_frames,
            manifest_out=manifest_out,
            splits_out="data/isl_splits.json",
            compat_splits=splits_out,
            dev_signers=("signer6",),
            test_signers=("signer7",)
        )
        return True

    if not os.path.isdir(videos_dir):
        print(f"[!] Videos directory not found: {videos_dir}")
        print("    Please place video clips into raw_data/ISL-CSLTR/videos/ or frames in Frames_Sentence_Level")
        return False
    if not os.path.isfile(labels_file):
        print(f"[!] Labels CSV not found: {labels_file}")
        print("    Please place labels CSV into raw_data/ISL-CSLTR/labels.csv")
        return False

    extensions = ['mp4', 'MOV', 'mov', 'avi', 'mkv']
    videos = [f for ext in extensions for f in glob.glob(os.path.join(videos_dir, '**', f'*.{ext}'), recursive=True)]
    if not videos:
        print(f"[!] No video files found in {videos_dir}")
        return False
    print(f"[*] Found {len(videos)} video files in {videos_dir}")

    # 1. Extract frames
    extract_frames_parallel(videos, videos_dir, frames_dir, size=size, workers=workers)

    # 2. Build manifest
    id_col, text_col, signer_col = auto_detect_isl_columns(labels_file)
    print(f"[*] Using CSV columns: id='{id_col}', text='{text_col}', signer='{signer_col or 'regex/unknown'}'")

    out_rows = []
    missing_frames = 0
    with open(labels_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            vid = os.path.splitext(os.path.basename(r[id_col]))[0]
            fd = os.path.join(frames_dir, vid)
            if not os.path.isdir(fd) or not os.listdir(fd):
                missing_frames += 1
                continue
            if signer_col and r.get(signer_col):
                signer = r[signer_col].strip()
            else:
                m = re.search(r'^(signer\d+|s\d+)', vid, re.IGNORECASE)
                signer = m.group(1).lower() if m else 'signer1'
            text = r[text_col].strip()
            out_rows.append(dict(id=vid, frame_dir=fd.replace('\\', '/'), signer=signer, text=text))

    os.makedirs(os.path.dirname(manifest_out) or '.', exist_ok=True)
    with open(manifest_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'frame_dir', 'signer', 'text'])
        w.writeheader()
        w.writerows(out_rows)

    all_signers = sorted({r['signer'] for r in out_rows})
    print(f"[✓] ISL-CSLTR manifest saved to {manifest_out}")
    print(f"    Total samples: {len(out_rows)} (Missing frames: {missing_frames})")
    print(f"    Signers detected ({len(all_signers)}): {all_signers}")

    # 3. Create splits
    os.makedirs(os.path.dirname(splits_out) or '.', exist_ok=True)
    if len(all_signers) >= 3:
        # Default: 1 signer for dev, 1 for test, rest for train
        dev_s = dev_signer or all_signers[1]
        test_s = test_signer or all_signers[-1]
        train_ids = [r['id'] for r in out_rows if r['signer'] not in (dev_s, test_s)]
        dev_ids = [r['id'] for r in out_rows if r['signer'] == dev_s]
        test_ids = [r['id'] for r in out_rows if r['signer'] == test_s]
        sp = dict(train=train_ids, dev=dev_ids, test=test_ids)
        print(f"[*] Signer-independent split: Train={len(train_ids)}, Dev ({dev_s})={len(dev_ids)}, Test ({test_s})={len(test_ids)}")
    else:
        # Fallback to random sentence split
        import random
        random.seed(3407)
        ids = [r['id'] for r in out_rows]
        random.shuffle(ids)
        n_test = max(1, int(len(ids) * 0.15))
        n_dev = max(1, int(len(ids) * 0.10))
        sp = dict(train=ids[n_test + n_dev:], dev=ids[:n_dev], test=ids[n_dev:n_test + n_dev])
        print(f"[*] Random split: Train={len(sp['train'])}, Dev={len(sp['dev'])}, Test={len(sp['test'])}")

    with open(splits_out, 'w', encoding='utf-8') as f:
        json.dump(sp, f, indent=2)
    print(f"[✓] Split saved to {splits_out}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Unified Preprocessing for GANet ISL")
    ap.add_argument('--dataset', choices=['include', 'isl', 'all'], default='all',
                    help="Which dataset to preprocess")
    ap.add_argument('--workers', type=int, default=4, help="Number of parallel extraction workers")
    ap.add_argument('--size', type=int, default=256, help="Target image frame size (size x size)")
    a = ap.parse_args()

    if a.dataset in ('include', 'all'):
        process_include(workers=a.workers, size=a.size)
        print()
    if a.dataset in ('isl', 'all'):
        process_isl(workers=a.workers, size=a.size)
        print()
    print("[✓] Preprocessing pipeline finished.")


if __name__ == '__main__':
    main()

