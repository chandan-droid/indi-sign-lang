"""Package project files and datasets for Google Colab training.

Creates clean zip archives ready to upload to Google Drive:
  1. ganet_code.zip: Codebase, configs, tools, manifests (no raw videos / extracted frames).
  2. include_frames.zip: Extracted frames in frames/include/ (if present).
  3. isl_frames.zip: Extracted frames in frames/isl/ (if present).

Usage:
  python tools/export_for_colab.py [--out_dir colab_package]
"""
import argparse
import os
import zipfile


def zip_directory(src_dir, zip_path, exclude_dirs=None):
    """Zip a directory recursively, skipping excluded subfolder names."""
    exclude_dirs = set(exclude_dirs or [])
    count = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(src_dir):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, src_dir)
                zf.write(full_path, rel_path)
                count += 1
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    return count, size_mb


def main():
    ap = argparse.ArgumentParser(description="Package GANet ISL for Google Colab")
    ap.add_argument('--out_dir', default='colab_package', help="Output directory for zip files")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    print("=" * 60)
    print("Packaging GANet ISL for Google Colab")
    print("=" * 60)

    # 1. Zip Code & Configs
    code_zip = os.path.join(a.out_dir, 'ganet_code.zip')
    print("[*] Creating ganet_code.zip (code, configs, tools, manifests)...")
    exclude = {'.git', 'raw_data', 'frames', 'work_dir', '__pycache__', '.ipynb_checkpoints', a.out_dir}
    n_files, mb = zip_directory('.', code_zip, exclude_dirs=exclude)
    print(f"    [✓] {code_zip}: {n_files} files ({mb:.2f} MB)")

    # 2. Zip INCLUDE frames (if extracted)
    if os.path.isdir('frames/include') and os.listdir('frames/include'):
        inc_zip = os.path.join(a.out_dir, 'include_frames.zip')
        print("[*] Creating include_frames.zip...")
        n_files, mb = zip_directory('frames/include', inc_zip)
        print(f"    [✓] {inc_zip}: {n_files} files ({mb:.2f} MB)")
    else:
        print("[!] No frames found in frames/include/ (skipping include_frames.zip)")

    # 3. Zip ISL frames (if extracted)
    if os.path.isdir('frames/isl') and os.listdir('frames/isl'):
        isl_zip = os.path.join(a.out_dir, 'isl_frames.zip')
        print("[*] Creating isl_frames.zip...")
        n_files, mb = zip_directory('frames/isl', isl_zip)
        print(f"    [✓] {isl_zip}: {n_files} files ({mb:.2f} MB)")
    else:
        print("[!] No frames found in frames/isl/ (skipping isl_frames.zip)")

    # 4. Zip Pre-trained INCLUDE Visual Model (if present)
    ckpt_src = 'work_dir/include_pretrain/include_visual.pt' if os.path.isfile('work_dir/include_pretrain/include_visual.pt') else 'work_dir/include_visual.pt'
    if os.path.isfile(ckpt_src):
        model_zip = os.path.join(a.out_dir, 'include_visual.zip')
        print(f"[*] Packaging pre-trained Stage 1 weights ({ckpt_src})...")
        with zipfile.ZipFile(model_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.write(ckpt_src, 'work_dir/include_pretrain/include_visual.pt')
        mb = os.path.getsize(model_zip) / (1024 * 1024)
        print(f"    [✓] {model_zip}: Stage 1 checkpoint ({mb:.2f} MB)")
    print("    Upload these zip files to Google Drive in a folder named 'GANet_ISL'.")


if __name__ == '__main__':
    main()

