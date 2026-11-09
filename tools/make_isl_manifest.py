"""Build the continuous manifest (id,frame_dir,signer,text) from a labels CSV.
Adapt --id_col/--text_col/--signer_col to your copy of ISL-CSLTR; --signer_regex extracts the signer from the id
or path if there is no signer column (group 1 of the regex).
Example: python tools/make_isl_manifest.py --labels labels.csv --frames frames --id_col video --text_col sentence \
         --signer_regex '^(signer\\d+)' --out data/isl_manifest.csv"""
import argparse, csv, os, re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True)
    ap.add_argument('--frames', required=True, help='root with one frame folder per video id')
    ap.add_argument('--id_col', default='video')
    ap.add_argument('--text_col', default='sentence')
    ap.add_argument('--signer_col', default=None)
    ap.add_argument('--signer_regex', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out, missing = [], 0
    for r in csv.DictReader(open(a.labels, encoding='utf-8-sig')):
        vid = os.path.splitext(os.path.basename(r[a.id_col]))[0]
        fd = os.path.join(a.frames, vid)
        if not os.path.isdir(fd):
            missing += 1
            continue
        if a.signer_col:
            signer = r[a.signer_col]
        elif a.signer_regex:
            m = re.search(a.signer_regex, r[a.id_col])
            signer = m.group(1) if m else 'unknown'
        else:
            signer = 'unknown'
        out.append(dict(id=vid, frame_dir=fd, signer=signer, text=r[a.text_col]))
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'frame_dir', 'signer', 'text'])
        w.writeheader()
        w.writerows(out)
    print(len(out), 'rows written;', missing, 'labels without frames;',
          'signers:', sorted({r['signer'] for r in out}))


if __name__ == '__main__':
    main()
