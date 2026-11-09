# Raw Data Directory Instructions

This folder holds the raw input datasets before frame extraction.

## 1. Directory Structure

Place your downloaded datasets as follows:

```text
raw_data/
├── INCLUDE/
│   ├── Adjectives/
│   │   ├── loud/
│   │   │   ├── clip_001.mp4
│   │   │   └── ...
│   │   └── ...
│   └── ...
│
└── ISL-CSLTR/
    ├── videos/
    │   ├── signer01_sample001.mp4
    │   ├── signer01_sample002.mp4
    │   └── ...
    └── labels.csv
```

### ISL-CSLTR `labels.csv` Format
The `labels.csv` should contain at least:
* A video identifier or filename column (e.g. `video`, `id`, `clip`)
* A sentence / gloss text column (e.g. `sentence`, `text`, `gloss`)
* (Optional) A signer column (e.g. `signer`, `speaker`)

Example:
```csv
video,signer,sentence
signer01_sample001,signer1,hello how are you
signer01_sample002,signer1,thank you very much
```

---

## 2. Preprocessing All Datasets

Run the master preprocessing pipeline:

```bash
# Preprocess everything in one go:
python tools/preprocess.py --all

# Or preprocess individually:
python tools/preprocess.py --dataset include --workers 8
python tools/preprocess.py --dataset isl --workers 8
```

---

## 3. Verification

After extraction, run the verification tool to check directory integrity, frame counts, vocabulary overlap, and CTC feasibility:

```bash
python tools/verify_data.py
```

