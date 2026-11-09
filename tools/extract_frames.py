"""Extract frames once (resized to 256x256 JPG) so training never decodes video.
Usage: python tools/extract_frames.py --videos DIR --out FRAMES_DIR [--ext mp4 MOV avi] [--every 1]
Each video  DIR/a/b/clip.mp4  becomes  FRAMES_DIR/a/b/clip/000000.jpg ..."""
import argparse, os, glob
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ext', nargs='+', default=['mp4', 'MOV', 'mov', 'avi', 'mkv'])
    ap.add_argument('--every', type=int, default=1, help='keep every k-th frame')
    ap.add_argument('--size', type=int, default=256)
    a = ap.parse_args()
    files = [f for e in a.ext for f in glob.glob(os.path.join(a.videos, '**', f'*.{e}'), recursive=True)]
    print(len(files), 'videos')
    for k, f in enumerate(sorted(files)):
        rel = os.path.splitext(os.path.relpath(f, a.videos))[0]
        out = os.path.join(a.out, rel)
        if os.path.isdir(out) and os.listdir(out):
            continue
        os.makedirs(out, exist_ok=True)
        cap, n, kept = cv2.VideoCapture(f), 0, 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if n % a.every == 0:
                cv2.imwrite(os.path.join(out, f'{kept:06d}.jpg'), cv2.resize(im, (a.size, a.size)),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                kept += 1
            n += 1
        cap.release()
        if kept == 0:
            print('WARNING: no frames read from', f)
        if k % 200 == 0:
            print(k, f, kept, 'frames', flush=True)


if __name__ == '__main__':
    main()
