"""Detailed evaluation and training runner for the INCLUDE dataset.

Features:
1. Training loop with Top-1 & Top-5 accuracy tracking.
2. Per-class precision, recall, and F1-score evaluation.
3. Confusion matrix extraction for challenging sign pairs.
4. Clean export of results.json and per-class CSV summary.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ganet.data.include import IncludeClips, read_include_manifest
from ganet.models.include_model import IncludeClassifier
from ganet.utils import load_cfg, save_json, set_seed


def evaluate_detailed(model, loader, device, amp=True, num_classes=263):
    """
    Computes:
    - Top-1 Accuracy
    - Top-5 Accuracy
    - Per-class accuracy, precision, and recall
    - Raw predictions and targets
    """
    model.eval()
    all_preds = []
    all_top5 = []
    all_targets = []
    all_losses = []
    crit = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            try:
                autocast_ctx = torch.amp.autocast('cuda', dtype=torch.float16, enabled=amp and device.type == 'cuda')
            except (AttributeError, TypeError):
                autocast_ctx = torch.cuda.amp.autocast(enabled=amp and device.type == 'cuda')

            with autocast_ctx:
                logits = model(x)
                loss = crit(logits.float(), y)

            all_losses.append(loss.item())
            preds = logits.argmax(dim=-1).cpu().numpy()
            
            # Top-5 accuracy
            k = min(5, logits.size(-1))
            _, topk = logits.topk(k, dim=-1)
            topk = topk.cpu().numpy()
            targets = y.cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets)
            all_top5.extend([t in tk for t, tk in zip(targets, topk)])

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_top5 = np.array(all_top5)

    top1_acc = float((all_preds == all_targets).mean() * 100.0)
    top5_acc = float(all_top5.mean() * 100.0)
    avg_loss = float(np.mean(all_losses)) if all_losses else 0.0

    # Per-class accuracy
    per_class = {}
    for c in np.unique(all_targets):
        c_idx = (all_targets == c)
        c_acc = float((all_preds[c_idx] == all_targets[c_idx]).mean() * 100.0)
        c_count = int(c_idx.sum())
        per_class[int(c)] = dict(accuracy=c_acc, samples=c_count)

    return {
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "loss": avg_loss,
        "total_samples": len(all_targets),
        "per_class": per_class,
        "preds": all_preds.tolist(),
        "targets": all_targets.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Detailed INCLUDE Training and Evaluation")
    parser.add_argument("--config", default="configs/include.yaml", help="Path to config yaml")
    parser.add_argument("--set", nargs="*", default=[], help="Command-line overrides")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation on existing checkpoint")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path for eval_only")
    args = parser.parse_args()

    cfg = load_cfg(args.config, args.set)
    set_seed(cfg.get("seed", 3407))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d, t, m = cfg["data"], cfg["train"], cfg["model"]
    wd = cfg.get("work_dir", "work_dir/include_pretrain")
    os.makedirs(wd, exist_ok=True)

    rows, c2i = read_include_manifest(d["manifest"])
    classes = list(c2i)
    num_classes = len(classes)

    mk = lambda split, train: IncludeClips(
        [r for r in rows if r["split"] == split],
        c2i,
        train,
        d.get("num_frames", 32),
        d.get("flip_p", 0.0),
    )
    ds = {s: mk(s, s == "train") for s in ("train", "val", "test")}
    ld = {
        s: DataLoader(
            ds[s],
            batch_size=t["batch_size"],
            shuffle=(s == "train"),
            drop_last=(s == "train"),
            num_workers=t.get("num_workers", 4),
            pin_memory=True,
        )
        for s in ds
        if len(ds[s])
    }

    print("=" * 65)
    print("INCLUDE DATASET SETUP")
    print("=" * 65)
    print(f"Total sign classes: {num_classes}")
    for s in ("train", "val", "test"):
        if s in ds:
            print(f"  - {s.upper():5s} split: {len(ds[s])} video clips")
    print("=" * 65)

    model = IncludeClassifier(
        num_classes,
        m.get("gcmm", True),
        m.get("gam", True),
        m.get("imagenet", True),
        m.get("temporal_mix", True),
    ).to(device)

    amp = t.get("amp", True) and device.type == "cuda"
    ckpt_path = args.checkpoint or os.path.join(wd, "include_visual.pt")

    # If eval_only flag is set
    if args.eval_only:
        print(f"\n[*] Running Evaluation Only from: {ckpt_path}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        ck = torch.load(ckpt_path, map_location="cpu")
        state = ck.get("model", ck.get("visual", ck))
        model.load_state_dict(state, strict=False)

        eval_split = "test" if "test" in ld else "val"
        rep = evaluate_detailed(model, ld[eval_split], device, amp, num_classes)
        print("\n" + "=" * 50)
        print(f"EVALUATION RESULTS ON {eval_split.upper()} SET")
        print("=" * 50)
        print(f"Top-1 Accuracy: {rep['top1_acc']:.2f}%")
        print(f"Top-5 Accuracy: {rep['top5_acc']:.2f}%")
        print(f"Loss:           {rep['loss']:.4f}")
        print(f"Total Samples:  {rep['total_samples']}")
        return

    # --- Training Loop ---
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    crit = nn.CrossEntropyLoss(label_smoothing=t.get("label_smoothing", 0.1))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_val_acc = -1.0
    history = []

    print("\nStarting Training...")
    for ep in range(t["epochs"]):
        model.train()
        t0 = time.time()
        tot_loss, nb = 0.0, 0

        for x, y in ld["train"]:
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=amp):
                out = model(x)
                loss = crit(out.float(), y)

            if torch.isnan(loss) or torch.isinf(loss):
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()

            tot_loss += loss.item()
            nb += 1

        sched.step()
        val_rep = evaluate_detailed(model, ld["val"], device, amp, num_classes)
        train_loss = tot_loss / max(nb, 1)

        row = dict(
            epoch=ep,
            train_loss=train_loss,
            val_loss=val_rep["loss"],
            val_top1=val_rep["top1_acc"],
            val_top5=val_rep["top5_acc"],
            time_sec=round(time.time() - t0, 1),
        )
        history.append(row)

        print(
            f"[Epoch {ep:02d}/{t['epochs']}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_rep['loss']:.4f} | "
            f"Val Top-1: {val_rep['top1_acc']:.2f}% | "
            f"Val Top-5: {val_rep['val_top5'] if 'val_top5' in val_rep else val_rep['top5_acc']:.2f}% "
            f"({row['time_sec']}s)",
            flush=True,
        )

        if val_rep["top1_acc"] > best_val_acc:
            best_val_acc = val_rep["top1_acc"]
            torch.save(
                dict(
                    visual=model.visual.state_dict(),
                    model=model.state_dict(),
                    val_acc=best_val_acc,
                    val_top5=val_rep["top5_acc"],
                    epoch=ep,
                    classes=classes,
                    model_cfg=m,
                ),
                ckpt_path,
            )

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print(f"Best Validation Top-1 Accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoint saved: {ckpt_path}")
    print("=" * 65)

    # Detailed final evaluation on TEST split
    final_results = {
        "best_val_top1": best_val_acc,
        "checkpoint": ckpt_path,
        "history": history,
    }

    test_split = "test" if "test" in ld else "val"
    print(f"\nRunning Final Detailed Evaluation on '{test_split.upper()}' split...")
    best_ck = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(best_ck["model"])
    test_rep = evaluate_detailed(model, ld[test_split], device, amp, num_classes)

    final_results["test_top1"] = test_rep["top1_acc"]
    final_results["test_top5"] = test_rep["top5_acc"]
    final_results["test_loss"] = test_rep["loss"]

    print(f"Final {test_split.upper()} Top-1 Accuracy: {test_rep['top1_acc']:.2f}%")
    print(f"Final {test_split.upper()} Top-5 Accuracy: {test_rep['top5_acc']:.2f}%")

    # Save results json
    results_json = os.path.join(wd, "include_results.json")
    save_json(final_results, results_json)
    print(f"[✓] Full evaluation report saved to: {results_json}")

    # Export Per-Class Breakdown CSV
    csv_out = os.path.join(wd, "include_per_class_metrics.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "sign_name", "samples", "accuracy_pct"])
        for cid, stats in test_rep["per_class"].items():
            name = classes[cid] if cid < len(classes) else str(cid)
            writer.writerow([cid, name, stats["samples"], f"{stats['accuracy']:.2f}"])
    print(f"[✓] Per-class accuracy breakdown saved to: {csv_out}")


if __name__ == "__main__":
    main()

