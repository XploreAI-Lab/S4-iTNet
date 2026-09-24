"""Evaluate a validation-selected checkpoint on window-level classification."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from config import ROOT, configure

CLASSES = ("CFSZ", "GNSZ", "ABSZ", "CTSZ")


def metrics(y, pred):
    cm = confusion_matrix(y, pred, labels=range(4))
    p, r, f, s = precision_recall_fscore_support(
        y, pred, labels=range(4), zero_division=0
    )
    return {
        "accuracy": float(np.trace(cm) / cm.sum()),
        "weighted_precision": float(np.average(p, weights=s)),
        "weighted_f1": float(np.average(f, weights=s)),
        "macro_f1": float(f.mean()),
        "balanced_accuracy": float(r[s > 0].mean()),
        "per_class_precision": p.tolist(),
        "per_class_recall": r.tolist(),
        "per_class_f1": f.tolist(),
        "support": s.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "checkpoints/s4_itnet_10s_wf1.pth"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--detection-only", action="store_true", help="Report channel detection metrics only (e.g. Stage 1 checkpoints)")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    config, _ = configure(args.config, args.device)
    from model.s4_itnet import S4iTNet

    torch.set_num_threads(min(8, torch.get_num_threads()))
    model = S4iTNet(config).to(args.device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    x = np.load(args.data_root / "test.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(args.data_root / "test_multi_label.npy", allow_pickle=False).astype(int)
    channel_targets = np.load(args.data_root / "test_bi_label.npy", allow_pickle=False)
    if channel_targets.shape != (len(y), 22) or not np.isin(channel_targets, [0, 1]).all():
        raise ValueError("Expected binary channel labels (N,22)")
    if x.shape != (len(y), 22, 2000) or not np.isin(y, range(4)).all():
        raise ValueError("Expected test EEG (N,22,2000) and four-class labels")
    args.output.mkdir(parents=True)
    predictions, confidences, weights = [], [], []
    channel_predictions = []
    started = time.perf_counter()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for start in range(0, len(y), args.batch_size):
            batch = torch.from_numpy(
                np.array(x[start : start + args.batch_size], dtype=np.float32)
            ).to(args.device)
            if not torch.isfinite(batch).all():
                raise ValueError("Non-finite input EEG")
            detection, logits, alpha = model.forward_with_alpha(batch)
            if not torch.isfinite(detection).all():
                raise ValueError("Non-finite detection probabilities")
            channel_predictions.extend((detection > 0.5).cpu().tolist())
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite classification logits")
            prob = logits.softmax(-1)
            predictions.extend(prob.argmax(-1).cpu().tolist())
            confidences.extend(prob.max(-1).values.cpu().tolist())
            weights.extend(alpha.cpu().tolist())
    result = {} if args.detection_only else metrics(y, predictions)
    channel_pred = np.asarray(channel_predictions, dtype=int).reshape(-1)
    channel_true = channel_targets.reshape(-1)
    pre, rec, f1, _ = precision_recall_fscore_support(
        channel_true, channel_pred, average="binary", zero_division=0
    )
    result["channel_detection"] = {
        "accuracy": float(np.mean(channel_true == channel_pred)),
        "precision": float(pre), "recall": float(rec), "f1": float(f1),
        "threshold": 0.5, "threshold_rule": "probability > threshold",
        "aggregation": "all window-channel pairs pooled",
        "confusion_matrix": confusion_matrix(channel_true, channel_pred, labels=[0, 1]).tolist(),
    }
    result.update(
        {
            "checkpoint_sha256": hashlib.sha256(
                args.checkpoint.read_bytes()
            ).hexdigest(),
            "elapsed_seconds": time.perf_counter() - started,
            "device": args.device,
            "parameters": sum(p.numel() for p in model.parameters()),
        }
    )
    if args.device.startswith("cuda"):
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2))
    np.save(args.output / "channel_weights.npy", np.array(weights))
    np.save(args.output / "channel_predictions.npy", np.asarray(channel_predictions, dtype=np.int8))
    if args.detection_only:
        print(json.dumps(result, indent=2))
        return
    with (args.output / "predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "array_index",
                "true_label",
                "true_class",
                "pred_label",
                "pred_class",
                "confidence",
            ]
        )
        w.writerows(
            (i, int(t), CLASSES[t], p, CLASSES[p], c)
            for i, (t, p, c) in enumerate(zip(y, predictions, confidences))
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
