"""Tools to create manifest and splits for RWTH-PHOENIX-Weather 2014T (Kaggle dataset).

Supports both:
1. Gzip/pickle annotations (.annotations_only.gzip / .pkl) containing lists of:
   {'name': 'train/11August_2010_Wednesday_tagesschau-1', 'signer': 'Signer08',
    'gloss': 'JETZT WETTER MORGEN DONNERSTAG ZWOELF FEBRUAR', 'text': '...'}
2. Standard RWTH .corpus / .csv files.

Maps video names directly to the .mp4 files or frame directories in /kaggle/input/.
"""
import argparse
import csv
import glob
import gzip
import json
import os
import pickle
import re


def load_annotation_file(file_path):
    """Loads either a gzip/pickle file or a plain text corpus."""
    if file_path.endswith(".gzip") or file_path.endswith(".gz") or file_path.endswith(".pkl") or file_path.endswith(".pickle"):
        opener = gzip.open if (file_path.endswith(".gzip") or file_path.endswith(".gz")) else open
        with opener(file_path, "rb") as f:
            data = pickle.load(f)
        return data
    return None


def index_video_files(search_root):
    """Indexes all .mp4 files by their base identifier."""
    print(f"[*] Indexing video/frame files under: {search_root}")
    video_map = {}
    for p in glob.glob(os.path.join(search_root, "**", "*.mp4"), recursive=True):
        basename = os.path.splitext(os.path.basename(p))[0]
        video_map[basename] = p.replace("\\", "/")
    print(f"[✓] Found {len(video_map)} .mp4 videos.")
    return video_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_root", type=str, default="/kaggle/input", help="Root search folder")
    parser.add_argument("--manifest_out", type=str, default="data/phoenix_manifest.csv")
    parser.add_argument("--splits_out", type=str, default="data/phoenix_splits.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.manifest_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.splits_out) or ".", exist_ok=True)

    # 1. Index all mp4 files
    video_map = index_video_files(args.search_root)

    # 2. Find annotation files
    all_files = glob.glob(os.path.join(args.search_root, "**", "*"), recursive=True)
    ann_files = [f for f in all_files if any(f.endswith(ext) for ext in [".gzip", ".gz", ".pkl", ".pickle"])]
    print(f"[*] Found {len(ann_files)} candidate annotation files.")

    splits = {"train": [], "dev": [], "test": []}
    manifest_rows = []
    matched_count = 0
    missing_video_count = 0

    for afile in ann_files:
        fname = os.path.basename(afile).lower()
        target_split = None
        if "train" in fname:
            target_split = "train"
        elif "dev" in fname or "val" in fname:
            target_split = "dev"
        elif "test" in fname:
            target_split = "test"

        if not target_split:
            continue

        print(f"[*] Loading {target_split} annotations from: {afile}")
        try:
            data = load_annotation_file(afile)
        except Exception as e:
            print(f"    [!] Error opening {afile}: {e}")
            continue

        if not isinstance(data, list):
            continue

        for item in data:
            name = item.get("name", "")
            signer = item.get("signer", "unknown")
            gloss = item.get("gloss", "")

            # Clean gloss text: uppercase, remove special punctuation
            clean_gloss = re.sub(r"[^\w\s]", " ", gloss).strip().upper()
            if not clean_gloss:
                continue

            # Strip split prefix if present, e.g. 'train/11August...' -> '11August...'
            base_id = os.path.basename(name)
            s_id = f"{target_split}_{base_id}"

            # Match to video path
            vid_path = video_map.get(base_id)
            if not vid_path:
                # Try finding without suffix or exact match
                for k, v in video_map.items():
                    if base_id in k or k in base_id:
                        vid_path = v
                        break

            if vid_path:
                manifest_rows.append({
                    "id": s_id,
                    "frame_dir": vid_path,
                    "signer": signer,
                    "text": clean_gloss,
                })
                splits[target_split].append(s_id)
                matched_count += 1
            else:
                missing_video_count += 1

    print(f"\n[✓] Matching complete: {matched_count} clips matched to videos.")
    if missing_video_count > 0:
        print(f"[!] Warning: {missing_video_count} annotations had no matching .mp4 video.")

    # Write manifest CSV
    with open(args.manifest_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "frame_dir", "signer", "text"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[✓] Saved manifest -> {args.manifest_out}")

    # Write splits JSON
    with open(args.splits_out, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"[✓] Saved splits -> {args.splits_out}")
    print(f"    Train: {len(splits['train'])} | Dev: {len(splits['dev'])} | Test: {len(splits['test'])}")


if __name__ == "__main__":
    main()
