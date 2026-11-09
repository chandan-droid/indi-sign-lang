The codebase is set up specifically for this pre-training and weight-transfer strategy, and the necessary wiring has been verified and updated so that it directly follows your modified architecture diagram.

---

### How the Strategy Works Under the Hood

```
   ┌──────────────────────────────────────────────────────────────┐
   │             Step 1: Pre-training on INCLUDE                  │
   │  Isolated Signs -> Stage 1 Visual (SMM + GCMM + GAM)        │
   │  -> Global Pooling -> Throwaway Classifier (num_classes)     │
   └──────────────────────────────┬───────────────────────────────┘
                                  │ Saves visual weights:
                                  │ "work_dir/include_pretrain/include_visual.pt"
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │           Step 2: Transfer into Modified Architecture        │
   │  • Loads: stem.*, layers.*, gcmm.*, gams.*                   │
   │  • Frozen for first 5 epochs (warm up new heads)            │
   │  • Unfrozen at lower lr (3e-5) for joint fine-tuning         │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │            Step 3: Training on Continuous ISL-CSLTR          │
   │  • Temporal Masking: drop chunks of consecutive frames       │
   │  • Mid-Stage Supervision: Linear VAC Head directly on Stage 1│
   │  • Stage 2 Dynamics: Conv1D -> Self-Attention -> BiLSTM      │
   │  • Auxiliary Task: GLP Head tapped directly from BiLSTM      │
   │  • Total Loss: L_Main(CTC) + L_GLP + alpha * L_VAC           │
   └──────────────────────────────────────────────────────────────┘
```

---

### What Was Updated in the Codebase

1. **GLP Tapped from BiLSTM ([`ganet.py`](file:///c:/My%20Projects/ganet/ganet_isl/ganet/models/ganet.py)):**
   - Added support for `glp_tap='temporal'`, which extracts features from the BiLSTM layer (`t['bilstm_feat']`) as shown by the purple arrow in your diagram.
2. **Temporal BiLSTM Feature Export ([`temporal.py`](file:///c:/My%20Projects/ganet/ganet_isl/ganet/models/temporal.py)):**
   - Updated `TemporalStage` to return `bilstm_feat=x` before classification.
3. **Mid-Stage VAC Tap on Stage 1 ([`ganet.py`](file:///c:/My%20Projects/ganet/ganet_isl/ganet/models/ganet.py)):**
   - Configured `vac_tap='visual'` so that the auxiliary VAC head connects directly after Stage 1 Visual Modeling (`f`), matching the orange dashed arrow in your diagram.
4. **Dedicated Configuration File ([`configs/isl_modified_ganet.yaml`](file:///c:/My%20Projects/ganet/ganet_isl/configs/isl_modified_ganet.yaml)):**
   - Created a ready-to-run configuration that activates:
     - `pretrained.path: work_dir/include_pretrain/include_visual.pt`
     - `data.mask_p: 0.5` (Temporal Masking)
     - `model.vac: true`, `model.vac_tap: visual`, `loss.vac: 1.0` (Mid-stage VAC)
     - `model.glp: true`, `model.glp_tap: temporal`, `loss.glp: 1.0` (BiLSTM GLP)
5. **Cross-Platform Shape Tests ([`tests/test_shapes.py`](file:///c:/My%20Projects/ganet/ganet_isl/tests/test_shapes.py)):**
   - Updated tests to verify pre-training weight extraction, loading into the modified architecture, and gradient backpropagation for all loss terms simultaneously.

---

### Step-by-Step Commands to Run

#### Step 1: Pre-train Stage 1 on INCLUDE
Pre-trains the visual representation (SMM, GCMM, and GAM) on isolated signs:
```bash
python train_include.py --config configs/include_pretrain.yaml
```
*Output:* Saves the visual stage checkpoint to `work_dir/include_pretrain/include_visual.pt`.

#### Step 2: Train the Modified Architecture on ISL-CSLTR
Initializes Stage 1 from the pre-trained weights, freezes it for 5 epochs to warm up the temporal and auxiliary heads, and then trains the full network end-to-end:
```bash
python train_isl.py --config configs/isl_modified_ganet.yaml
```
During training:
* `[pretrained] loaded 122/122 visual tensors from work_dir/include_pretrain/include_visual.pt`
* Epochs 0–4: Stage 1 is frozen; only Temporal, GLP, and VAC heads are trained.
* Epochs 5–59: All weights are unfrozen and updated simultaneously with the total loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Main}} + \lambda_{\text{GLP}} \mathcal{L}_{\text{GLP}} + \alpha \mathcal{L}_{\text{VAC}}$$

#### Step 3: Evaluate on the Test Split
Evaluates the best checkpoint using CTC beam search decoding:
```bash
python evaluate.py --ckpt work_dir/isl_modified_ganet_s3407/best.pt --split test --beam 10
```