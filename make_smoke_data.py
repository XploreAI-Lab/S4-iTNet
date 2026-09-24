"""Generate synthetic EEG-shaped fixtures, with one sample per class per split."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def generate(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2023)
    rows = []
    t = np.arange(2000) / 200
    for split in ("train", "val", "test"):
        x = rng.normal(0, 5, (4, 22, 2000)).astype("float32")
        labels = np.arange(4, dtype="int32")
        binary = np.zeros((4, 22), dtype="int32")
        for cls in range(4):
            binary[cls, : cls + 2] = 1
            x[cls, : cls + 2] += 15 * np.sin(2 * np.pi * (cls + 3) * t)
            rows.append(
                {
                    "split": split,
                    "array_index": cls,
                    "event_id": f"synthetic_{split}_{cls}",
                    "patient_id": f"synthetic_{split}_{cls}",
                    "class_id": cls,
                }
            )
        np.save(output / f"{split}.npy", x)
        np.save(output / f"{split}_multi_label.npy", labels)
        np.save(output / f"{split}_bi_label.npy", binary)
    with (output / "window_manifest_array_order.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (output / "SYNTHETIC_DATA.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "clinical_data": False,
                "purpose": "Shape, label and pipeline checks only; not scientific performance evaluation",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset/smoke_demo",
    )
    generate(p.parse_args().output)
