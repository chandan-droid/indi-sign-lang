"""Verification and sanity-check tool for GANet ISL datasets.

Verifies:
  1. INCLUDE manifest and extracted frame integrity.
  2. ISL-CSLTR manifest and frame counts.
  3. CTC feasibility: checks if video_len / (2^num_pool) >= label_length for every sample.
  4. Vocabulary coverage: checks if dev/test sets contain unseen words that CTC cannot predict.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict


def tokenize(text):
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()


def verify_include(manifest_path="data/include_manifest.csv"):
    print("=" * 60)
    print("Verifying INCLUDE Dataset")
    print("=" * 60)
    if not os.path.isfile(manifest_path):
        print(f"[✗] Manifest not found: {manifest_path}")
        return False
    
    with open(manifest_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    
    if not rows:
        print(f"[✗] Manifest is empty: {manifest_path}")
        return False
    
    missing_dirs = 0
    empty_dirs = 0
    splits_count = defaultdict(int)
    classes = set()
    frame_counts = []
    
    for r in rows:
        splits_count[r['split']] += 1
        classes.add(r['label'])
        fd = r['frame_dir']
        if not os.path.isdir(fd):
            missing_dirs += 1
        else:
            n_frames = len(glob.glob(os.path.join(fd, '*.jpg'))) + len(glob.glob(os.path.join(fd, '*.png')))
            if n_frames == 0:
                empty_dirs += 1
            else:
                frame_counts.append(n_frames)
                
    print(f"Total samples : {len(rows)}")
    print(f"Total classes : {len(classes)}")
    print(f"Splits        : {dict(splits_count)}")
    if frame_counts:
        print(f"Frame counts  : min={min(frame_counts)}, max={max(frame_counts)}, avg={sum(frame_counts)/len(frame_counts):.1f}")
    if missing_dirs > 0:
        print(f"[!] Warning: {missing_dirs} directories in manifest do not exist on disk.")
    if empty_dirs > 0:
        print(f"[!] Warning: {empty_dirs} directories contain 0 frames.")
    
    ok = (missing_dirs == 0 and empty_dirs == 0)
    print(f"Status        : {'[✓] READY FOR PRE-TRAINING' if ok else '[✗] ISSUES DETECTED'}")
    return ok


def verify_isl(manifest_path="data/isl_manifest.csv", splits_path="splits/isl_signer_v1.json", num_pool=2):
    print("=" * 60)
    print("Verifying ISL-CSLTR Dataset")
    print("=" * 60)
    if not os.path.isfile(manifest_path):
        print(f"[✗] Manifest not found: {manifest_path}")
        return False
    if not os.path.isfile(splits_path):
        print(f"[✗] Splits file not found: {splits_path}")
        return False
        
    with open(manifest_path, encoding='utf-8') as f:
        rows = {r['id']: r for r in csv.DictReader(f)}
    
    with open(splits_path, encoding='utf-8') as f:
        splits = json.load(f)
        
    downsample = 2 ** num_pool
    print(f"Total samples in manifest : {len(rows)}")
    print(f"Temporal pool factor      : 2^{num_pool} = {downsample}x downsampling")
    
    train_tokens = set()
    ctc_infeasible = []
    missing_dirs = 0
    
    for split_name in ('train', 'dev', 'test'):
        ids = splits.get(split_name, [])
        signers = set()
        split_frames = []
        for vid in ids:
            if vid not in rows:
                print(f"[!] Split ID {vid} not found in manifest!")
                continue
            r = rows[vid]
            signers.add(r['signer'])
            toks = tokenize(r['text'])
            if split_name == 'train':
                train_tokens.update(toks)
                
            fd = r['frame_dir']
            if not os.path.isdir(fd):
                missing_dirs += 1
                n_frames = 0
            else:
                n_frames = len(glob.glob(os.path.join(fd, '*.jpg'))) + len(glob.glob(os.path.join(fd, '*.png')))
                split_frames.append(n_frames)
            
            # Check CTC condition: out_len = frames // downsample must be >= len(toks)
            out_len = n_frames // downsample
            if out_len < len(toks) and n_frames > 0:
                ctc_infeasible.append((vid, split_name, n_frames, out_len, len(toks), r['text']))
                
        print(f"Split '{split_name:5s}' : {len(ids)} clips, Signers: {sorted(signers)}")
        if split_frames:
            print(f"                Frames: min={min(split_frames)}, max={max(split_frames)}, avg={sum(split_frames)/len(split_frames):.1f}")
            
    # Check vocabulary coverage
    print("-" * 60)
    print(f"Train vocabulary size     : {len(train_tokens)} unique glosses")
    for s in ('dev', 'test'):
        s_tokens = {w for vid in splits.get(s, []) if vid in rows for w in tokenize(rows[vid]['text'])}
        unseen = s_tokens - train_tokens
        if unseen:
            print(f"[!] Warning: {len(unseen)} words in '{s}' were never seen in 'train' split!")
            print(f"    Sample unseen words: {list(unseen)[:5]}")
        else:
            print(f"[✓] Zero vocabulary leakage in '{s}' split (all words seen in train)")

    # Report CTC feasibility
    print("-" * 60)
    if ctc_infeasible:
        print(f"[!] WARNING: {len(ctc_infeasible)} samples have fewer frames than required for CTC!")
        print(f"    (Frame length // {downsample} < label token count)")
        for item in ctc_infeasible[:3]:
            print(f"    Clip: {item[0]} [{item[1]}] -> {item[2]} frames // {downsample} = {item[3]} < {item[4]} words ('{item[5]}')")
        print(f"    RECOMMENDATION: Set 'model.num_pool: 1' (or 0) in configs/isl_modified_ganet.yaml")
    else:
        print(f"[✓] All samples satisfy CTC sequence length constraints (num_pool={num_pool})")
        
    ok = (missing_dirs == 0 and len(ctc_infeasible) == 0)
    print(f"Status                    : {'[✓] READY FOR TRAINING' if ok else '[!] ATTENTION NEEDED'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Verify datasets and manifests for GANet ISL")
    ap.add_argument('--include_manifest', default="data/include_manifest.csv")
    ap.add_argument('--isl_manifest', default="data/isl_manifest.csv")
    ap.add_argument('--splits', default="splits/isl_signer_v1.json")
    ap.add_argument('--num_pool', type=int, default=2)
    a = ap.parse_args()

    inc_ok = verify_include(a.include_manifest)
    print()
    isl_ok = verify_isl(a.isl_manifest, a.splits, a.num_pool)


if __name__ == '__main__':
    main()

