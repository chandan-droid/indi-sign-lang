"""Generate lightweight synthetic dummy data for testing the GANet ISL pipeline.

Creates:
  1. frames/include/ (5 classes, 4 clips each, 32 frames of 256x256)
  2. data/include_manifest.csv
  3. raw_data/ISL-CSLTR/labels.csv
  4. frames/isl/ (12 sentence clips across 3 signers, 64 frames of 256x256)
  5. data/isl_manifest.csv
  6. splits/isl_signer_v1.json

Usage:
  python tools/create_dummy_data.py
"""
import csv
import json
import os
import numpy as np
from PIL import Image


def create_frames(folder, n_frames=32, size=256, base_color=(120, 100, 150)):
    os.makedirs(folder, exist_ok=True)
    for i in range(n_frames):
        # Slightly varying color across time to mimic motion
        r = (base_color[0] + i * 2) % 256
        g = (base_color[1] + i * 3) % 256
        b = (base_color[2] + i * 4) % 256
        img_arr = np.zeros((size, size, 3), dtype=np.uint8)
        img_arr[:, :] = [r, g, b]
        # Draw a simple moving square
        sq_size = 40
        x = int(20 + (i / max(1, n_frames)) * (size - sq_size - 40))
        y = int(size // 2 - sq_size // 2)
        img_arr[y:y+sq_size, x:x+sq_size] = [255 - r, 255 - g, 255 - b]
        Image.fromarray(img_arr).save(os.path.join(folder, f"{i:06d}.jpg"), quality=90)


def generate_dummy_include(out_frames="frames/include", out_manifest="data/include_manifest.csv"):
    print("[*] Generating dummy INCLUDE dataset...")
    classes = ["hello", "thank_you", "welcome", "good_morning", "help"]
    rows = []
    for cls in classes:
        for clip_id in range(1, 5):
            clip_name = f"clip_{clip_id:03d}"
            folder = os.path.join(out_frames, cls, clip_name)
            create_frames(folder, n_frames=32, base_color=(50 * clip_id, 30 * clip_id, 40))
            split = "train" if clip_id <= 2 else ("val" if clip_id == 3 else "test")
            rows.append(dict(frame_dir=os.path.abspath(folder), label=cls, split=split))
    
    os.makedirs(os.path.dirname(out_manifest) or '.', exist_ok=True)
    with open(out_manifest, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['frame_dir', 'label', 'split'])
        w.writeheader()
        w.writerows(rows)
    print(f"[✓] Created {len(rows)} dummy INCLUDE clips in {out_frames} & manifest {out_manifest}")


def generate_dummy_isl(out_raw="raw_data/ISL-CSLTR", out_frames="frames/isl",
                       out_manifest="data/isl_manifest.csv", out_splits="splits/isl_signer_v1.json"):
    print("[*] Generating dummy ISL-CSLTR continuous dataset...")
    sentences = [
        "hello how are you",
        "good morning welcome",
        "thank you very much",
        "i need your help",
        "where is the hospital",
        "what is your name",
        "nice to meet you",
        "please can you help me",
        "see you tomorrow again",
        "i am learning sign language",
        "today weather is good",
        "take care of yourself"
    ]
    signers = ["signer1", "signer2", "signer3"]
    labels_rows = []
    manifest_rows = []
    
    os.makedirs(out_raw, exist_ok=True)
    labels_path = os.path.join(out_raw, "labels.csv")
    
    train_ids, dev_ids, test_ids = [], [], []
    
    for idx, sent in enumerate(sentences):
        signer = signers[idx % len(signers)]
        vid = f"{signer}_sample_{idx+1:03d}"
        folder = os.path.join(out_frames, vid)
        # Ensure enough frames for CTC: e.g. 64 frames (64 // 4 = 16 tokens > sentence length)
        create_frames(folder, n_frames=64, base_color=(20 * (idx+1), 40, 80))
        
        labels_rows.append(dict(video=vid, signer=signer, sentence=sent))
        manifest_rows.append(dict(id=vid, frame_dir=os.path.abspath(folder), signer=signer, text=sent))
        
        if signer == "signer1":
            train_ids.append(vid)
        elif signer == "signer2":
            dev_ids.append(vid)
        else:
            test_ids.append(vid)
            
    # Write labels.csv
    with open(labels_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['video', 'signer', 'sentence'])
        w.writeheader()
        w.writerows(labels_rows)
        
    # Write manifest.csv
    os.makedirs(os.path.dirname(out_manifest) or '.', exist_ok=True)
    with open(out_manifest, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'frame_dir', 'signer', 'text'])
        w.writeheader()
        w.writerows(manifest_rows)
        
    # Write splits.json
    os.makedirs(os.path.dirname(out_splits) or '.', exist_ok=True)
    with open(out_splits, 'w', encoding='utf-8') as f:
        json.dump(dict(train=train_ids, dev=dev_ids, test=test_ids), f, indent=2)
        
    print(f"[✓] Created {len(manifest_rows)} dummy ISL samples in {out_frames}")
    print(f"[✓] Created {labels_path}, {out_manifest}, and {out_splits}")


def main():
    generate_dummy_include()
    print()
    generate_dummy_isl()
    print("\n[✓] Dummy dataset generation complete!")
    print("    You can now run: python tools/verify_data.py")


if __name__ == '__main__':
    main()
