"""End-to-end EDF and typed-annotation preprocessing with artificial recordings."""

import csv
import subprocess
import sys

import numpy as np
from pyedflib import highlevel

from data_audit import audit
from preprocessing.edf_io import TCP_CHANNELS


def test_synthetic_edf_to_arrays(tmp_path):
    raw = tmp_path / "raw"
    rows = []
    electrodes = sorted({e for a, b, _ in TCP_CHANNELS for e in (a, b)})
    rng = np.random.default_rng(77)
    for source, split in [("train", "train"), ("dev", "val"), ("eval", "test")]:
        for cls, label in enumerate(["fnsz", "gnsz", "absz", "tnsz"]):
            patient = f"synthetic_{split}_{cls}"
            record = raw / "edf" / source / patient / "session" / "record.edf"
            record.parent.mkdir(parents=True)
            headers = highlevel.make_signal_headers(
                [f"EEG {e}-REF" for e in electrodes],
                sample_frequency=200,
                physical_min=-200,
                physical_max=200,
            )
            highlevel.write_edf(
                str(record), rng.normal(0, 10, (len(electrodes), 6000)), headers
            )
            record.with_suffix(".csv").write_text(
                "channel,start_time,stop_time,label,confidence\nFP1-F7,10,20,"
                + label
                + ",1\n"
            )
            record.with_suffix(".csv_bi").write_text(
                "channel,start_time,stop_time,label,confidence\nTERM,10,20,seiz,1\n"
            )
            rows.append((patient, split))
    assignment = tmp_path / "split.csv"
    with assignment.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "split"])
        writer.writerows(rows)
    out = tmp_path / "processed"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "preprocessing.build_dataset",
            "--input-root",
            str(raw),
            "--output-dir",
            str(out),
            "--duration-sec",
            "10",
            "--assignment-csv",
            str(assignment),
            "--workers",
            "1",
            "--min-free-gb",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    checked = audit(out)
    assert checked["attribution_verified"]
    for split in ("train", "val", "test"):
        assert all(n > 0 for n in checked["splits"][split]["class_counts"])
