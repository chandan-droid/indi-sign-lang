"""Scan ISL-CSLRT pre-extracted frames (Frames_Sentence_Level) and generate manifest + signer-independent splits.

Dataset structure:
  raw_data/ISL-CSLTR/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Sentence_Level/<Sentence_Text>/<Signer_ID>/*.jpg
Output:
  data/isl_manifest.csv (id, frame_dir, signer, text)
  data/isl_splits.json  (train: signers 1-5, dev: signer 6, test: signer 7)
  splits/isl_signer_v1.json (same split for compatibility)
"""
import argparse
import glob
import json
import os
import re
import pandas as pd


def clean_sentence(text):
    """Normalize sentence text."""
    text = text.strip()
    # Remove leading numbering like '1.', '01 - ', etc. if present
    text = re.sub(r'^\d+[\.\-_ ]+', '', text)
    return text


def find_sentence_frames_dir(search_root="raw_data/ISL-CSLTR"):
    """Locate the Frames_Sentence_Level directory automatically."""
    for root, dirs, files in os.walk(search_root):
        if os.path.basename(root).lower() == 'frames_sentence_level':
            return root
    return None


def generate_manifest_and_splits(frames_root,
                                 manifest_out="data/isl_manifest.csv",
                                 splits_out="data/isl_splits.json",
                                 compat_splits="splits/isl_signer_v1.json",
                                 dev_signers=("signer6",),
                                 test_signers=("signer7",)):
    print(f"[*] Scanning pre-extracted frames in: {frames_root}")

    # Gather leaf frame directories
    leaf_dirs = []
    for root, dirs, files in os.walk(frames_root):
        has_imgs = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files)
        if has_imgs:
            leaf_dirs.append(root)

    leaf_dirs = sorted(leaf_dirs)
    print(f"[*] Found {len(leaf_dirs)} leaf frame directories.")
    if not leaf_dirs:
        raise RuntimeError(f"No image frames found under {frames_root}!")

    rows = []
    for idx, d in enumerate(leaf_dirs):
        signer_tag = os.path.basename(d).strip()
        sentence_dir = os.path.dirname(d)
        sentence_text = clean_sentence(os.path.basename(sentence_dir))

        # Format signer name uniformly: '1' -> 'signer1', 'signer1' -> 'signer1'
        if signer_tag.isdigit():
            signer = f"signer{signer_tag}"
        elif signer_tag.lower().startswith('signer'):
            signer = signer_tag.lower()
        else:
            signer = f"signer_{signer_tag.lower()}"

        # Clean relative path with forward slashes
        rel_path = os.path.relpath(d, ".").replace("\\", "/")
        sample_id = f"{signer}_sample_{idx+1:04d}"

        rows.append({
            "id": sample_id,
            "frame_dir": rel_path,
            "signer": signer,
            "text": sentence_text
        })

    # Save manifest CSV
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(manifest_out) or ".", exist_ok=True)
    df.to_csv(manifest_out, index=False, encoding='utf-8')
    print(f"[✓] Saved {len(df)} samples to {manifest_out}")

    # Inspect signers
    all_signers = sorted(df['signer'].unique().tolist())
    print(f"[*] Detected signers ({len(all_signers)}): {all_signers}")

    # Build Signer-Independent split
    train_ids = []
    dev_ids = []
    test_ids = []

    for _, r in df.iterrows():
        sid = r['signer']
        if sid in test_signers or (len(all_signers) >= 3 and sid == all_signers[-1] and not test_signers):
            test_ids.append(r['id'])
        elif sid in dev_signers or (len(all_signers) >= 3 and sid == all_signers[-2] and not dev_signers):
            dev_ids.append(r['id'])
        else:
            train_ids.append(r['id'])

    # If any split is empty, fallback to 70/15/15
    if not dev_ids or not test_ids:
        print("[!] Warning: specified signers not matched; falling back to last 2 signers for dev/test.")
        train_ids = [r['id'] for _, r in df.iterrows() if r['signer'] not in all_signers[-2:]]
        dev_ids = [r['id'] for _, r in df.iterrows() if r['signer'] == all_signers[-2]]
        test_ids = [r['id'] for _, r in df.iterrows() if r['signer'] == all_signers[-1]]

    splits = {
        "train": train_ids,
        "dev": dev_ids,
        "test": test_ids
    }

    # Verify zero vocabulary leakage
    train_texts = df[df['id'].isin(train_ids)]['text']
    train_words = {w for t in train_texts for w in t.lower().split()}
    for s_name, s_ids in [("dev", dev_ids), ("test", test_ids)]:
        s_texts = df[df['id'].isin(s_ids)]['text']
        s_words = {w for t in s_texts for w in t.lower().split()}
        oov = s_words - train_words
        if oov:
            print(f"[!] Warning: {len(oov)} OOV words in {s_name}: {list(oov)[:5]}")
        else:
            print(f"[✓] Zero vocabulary leakage in {s_name}! (100% of words seen in train)")

    # Save split JSONs
    for out_p in [splits_out, compat_splits]:
        os.makedirs(os.path.dirname(out_p) or ".", exist_ok=True)
        with open(out_p, 'w', encoding='utf-8') as f:
            json.dump(splits, f, indent=2)
        print(f"[✓] Saved split to {out_p}: train={len(train_ids)}, dev={len(dev_ids)}, test={len(test_ids)}")

    return df, splits


def main():
    ap = argparse.ArgumentParser(description="Generate ISL-CSLRT manifest and splits from Frames_Sentence_Level")
    ap.add_argument("--search_root", default="raw_data/ISL-CSLTR", help="Directory to search for Frames_Sentence_Level")
    ap.add_argument("--manifest_out", default="data/isl_manifest.csv")
    ap.add_argument("--splits_out", default="data/isl_splits.json")
    ap.add_argument("--compat_splits", default="splits/isl_signer_v1.json")
    ap.add_argument("--dev_signer", default="signer6")
    ap.add_argument("--test_signer", default="signer7")
    a = ap.parse_args()

    frames_root = find_sentence_frames_dir(a.search_root)
    if not frames_root:
        # Check standard location directly
        std_path = os.path.join(a.search_root, "ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Sentence_Level")
        if os.path.isdir(std_path):
            frames_root = std_path
        else:
            print(f"[!] Frames_Sentence_Level not found in {a.search_root}")
            return

    generate_manifest_and_splits(
        frames_root,
        manifest_out=a.manifest_out,
        splits_out=a.splits_out,
        compat_splits=a.compat_splits,
        dev_signers=(a.dev_signer,),
        test_signers=(a.test_signer,)
    )


if __name__ == "__main__":
    main()

