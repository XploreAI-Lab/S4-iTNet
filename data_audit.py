"""Validate preprocessed arrays and their manifest."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def audit(root, splits=("train", "val", "test")):
    root = Path(root)
    manifest = root / "window_manifest_array_order.csv"
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"split", "array_index", "event_id", "class_id", "patient_id"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Manifest requires {sorted(required)}")
        rows = list(reader)
    report = {"splits": {}}
    patients, events, event_class = {}, {}, {}
    for split in splits:
        x = np.load(root / f"{split}.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(root / f"{split}_multi_label.npy", allow_pickle=False)
        b = np.load(root / f"{split}_bi_label.npy", mmap_mode="r", allow_pickle=False)
        if x.shape != (len(y), 22, 2000) or b.shape != (len(y), 22) or y.ndim != 1:
            raise ValueError(
                f"Invalid shapes in {split}: {x.shape}, {y.shape}, {b.shape}"
            )
        if not np.isin(y, range(4)).all() or not np.isin(b, [0, 1]).all():
            raise ValueError("Invalid class or binary label")
        for start in range(0, len(y), 64):
            if not np.isfinite(x[start : start + 64]).all():
                raise ValueError(f"Non-finite EEG in {split} at {start}")
        selected = [r for r in rows if r["split"] == split]
        indices = [int(r["array_index"]) for r in selected]
        if len(selected) != len(y) or sorted(indices) != list(range(len(y))):
            raise ValueError(f"Missing, duplicate or invalid array indices in {split}")
        ordered = sorted(selected, key=lambda r: int(r["array_index"]))
        mismatches = sum(int(r["class_id"]) != int(y[i]) for i, r in enumerate(ordered))
        if mismatches:
            raise ValueError(f"{split}: {mismatches} manifest/label disagreements")
        for r in ordered:
            if not r["patient_id"] or not r["event_id"]:
                raise ValueError("Empty patient/event identity")
            for seen, key in ((patients, r["patient_id"]), (events, r["event_id"])):
                if key in seen and seen[key] != split:
                    raise ValueError(f"Identity crosses splits: {key}")
                seen[key] = split
            if r["event_id"] in event_class and event_class[r["event_id"]] != int(
                r["class_id"]
            ):
                raise ValueError("An event has conflicting classes")
            event_class[r["event_id"]] = int(r["class_id"])
        report["splits"][split] = {
            "windows": len(y),
            "class_counts": np.bincount(y.astype(int), minlength=4).tolist(),
            "manifest_label_mismatches": mismatches,
        }
    report["attribution_verified"] = True
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(audit(a.data_root), indent=2))
