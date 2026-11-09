# GANet-style CSLR for Indian Sign Language (reimplementation)

**Status: independent reimplementation from the paper text (Chu et al., Complex Eng. Syst. 2026, 6, 7).**
No official GANet code was found; the paper only links the EvCSLR dataset repo. GCMM, GAM and the temporal stage follow
the paper's description and figures, but internal details the paper does not give (GAM gating, GCMM temporal conv) are
my interpretation and are marked in the docstrings. Do not expect to match the paper's numbers exactly.

**Tested here:** pure-python parts (WER, CTC decoders, augmentations, split tool): `python tests/test_pure.py`.
**NOT run here (no PyTorch/GPU in the build sandbox):** all model, data-loading and training code. Syntax-checked only.
Run `python tests/test_shapes.py` first on your machine; fix anything it reports before training.

## Layout
```
ganet/models/   gcmm.py gam.py heads.py visual.py temporal.py ganet.py include_model.py
ganet/data/     common.py augment.py include.py isl_csltr.py
ganet/          losses.py engine.py metrics.py utils.py
tools/          extract_frames.py make_isl_manifest.py make_splits.py make_include_manifest.py
configs/        include_pretrain.yaml isl_baseline.yaml isl_ganet.yaml isl_ganet_pretrained.yaml
train_include.py  train_isl.py  evaluate.py  tests/
```

## Setup
```bash
pip install -r requirements.txt
python tests/test_pure.py && python tests/test_shapes.py
```

## 1. Data preparation
```bash
# frames (once; resized to 256x256 JPG)
python tools/extract_frames.py --videos ISL-CSLTR/videos --out frames/isl
python tools/extract_frames.py --videos INCLUDE --out frames/include

# ISL-CSLTR manifest + frozen splits (adapt column names/regex to your copy)
python tools/make_isl_manifest.py --labels ISL-CSLTR/labels.csv --frames frames/isl \
       --id_col video --text_col sentence --signer_regex '^(signer\d+)' --out data/isl_manifest.csv
python tools/make_splits.py --manifest data/isl_manifest.csv --mode signer --dev S2 --test S7 --out splits/isl_signer_v1.json
python tools/make_splits.py --manifest data/isl_manifest.csv --mode loso --out_dir splits/      # 7 folds

# INCLUDE manifest (use --train_list/--test_list for the official split)
python tools/make_include_manifest.py --frames frames/include --out data/include_manifest.csv
```
Frames per clip matter: if ISL-CSLTR clips are short, set `model.num_pool` to 1 or 0 in the config
(output length = frames / 2^num_pool must be >= label length for CTC).

## 2. Pre-train on INCLUDE, then fine-tune on ISL-CSLTR
```bash
python train_include.py --config configs/include_pretrain.yaml
python train_isl.py --config configs/isl_baseline.yaml            # E0 control
python train_isl.py --config configs/isl_ganet.yaml               # E1 GANet-style from ImageNet
python train_isl.py --config configs/isl_ganet_pretrained.yaml    # E2 with INCLUDE weights
python train_isl.py --config configs/isl_ganet_pretrained.yaml --set pretrained.transfer_gam=false work_dir=work_dir/isl_inc_nogam
python train_isl.py --config configs/isl_modified_ganet.yaml          # E5: Full modified architecture
python evaluate.py --ckpt work_dir/isl_modified_ganet_s3407/best.pt --split test --beam 10
```
Repeat with `--set seed=1 work_dir=...` for 3 seeds. Model selection uses the dev split only; the test split is evaluated once at the end.

## Experiment matrix
| ID | Config |
|---|---|
| E0 | isl_baseline |
| E1 | isl_ganet |
| E2 | isl_ganet_pretrained |
| E3 | E1 + `data.mask_p=0.5` |
| E4 | E1 + `model.vac=true loss.vac=1.0` (+ `loss.kd=25.0` for full VAC) |
| E5 | isl_modified_ganet (E2 + Temporal Masking + Mid-stage VAC + BiLSTM-tapped GLP) |

## Transfer contract
INCLUDE and ISL-CSLTR models share `VisualStage1`, so checkpoint keys are identical (`stem.*`, `layers.*`, `gcmm.*`, `gams.*`).
`load_pretrained_visual` prints how many tensors loaded and which were skipped. The temporal stage, GLP head and VAC head are
never transferred (different label spaces).

## Known limitations / things to verify
- Padded frames in a batch pass through the temporal convolutions; use batch size 2 and similar-length clips, or add masking.
- GAM/GCMM internals are interpretations (see docstrings). Ablate them rather than trusting them.
- The paper's learning-rate schedule (decay at epochs 10 and 20 of 100) looks unusual; configs default to 30/45 of 60. Tune on dev.
- Grad-CAM and the hand-mask focus metric are not included yet.
- Distillation (`loss.kd`) needs `vac_tap: conv`; with `vac_tap: visual` only the auxiliary CTC is used.
