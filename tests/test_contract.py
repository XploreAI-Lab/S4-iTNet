"""Regression checks for configuration, sampling, and data contracts."""

import csv

import numpy as np
import pytest

from config import configure
from data_audit import audit
from make_smoke_data import generate


def test_config_and_sampler():
    args, cfg = configure()
    assert (args.d_model, args.n_layers, args.e_layers, args.n_heads) == (256, 4, 4, 8)
    assert args.epoch_class_targets == [8000, 4000, 800, 800]
    from data import ClassEpochSampler

    labels = np.repeat(np.arange(4), 10)
    a = ClassEpochSampler(labels, {0: 5, 1: 4, 2: 3, 3: 2}, seed=11)
    b = ClassEpochSampler(labels, {0: 5, 1: 4, 2: 3, 3: 2}, seed=11)
    first = list(a)
    assert first == list(b)
    assert len(set(first)) == 14
    assert np.bincount(labels[first]).tolist() == [5, 4, 3, 2]
    assert list(a) != first


def test_data_audit_rejects_bad_manifest(tmp_path):
    root = tmp_path / "data"
    generate(root)
    assert audit(root)["attribution_verified"]
    p = root / "window_manifest_array_order.csv"
    rows = list(csv.DictReader(p.open()))
    rows[0]["class_id"] = "3"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match="disagreements"):
        audit(root)


def test_data_audit_rejects_patient_overlap(tmp_path):
    root = tmp_path / "data"
    generate(root)
    p = root / "window_manifest_array_order.csv"
    rows = list(csv.DictReader(p.open()))
    rows[4]["patient_id"] = rows[0]["patient_id"]
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match="crosses splits"):
        audit(root)


def test_preprocessing_window_rule():
    from preprocessing.build_dataset import (
        SeizureEvent,
        candidate_starts,
        overlap_seconds,
    )

    ev = SeizureEvent(
        "e", "train", "p", "synthetic.edf", 60.0, 20.0, 30.0, 0, "CFSZ", ("fnsz",)
    )
    starts = candidate_starts(ev, 10.0, 0.4, 10.0, 200)
    assert starts
    assert all(
        overlap_seconds(s, s + 10.0, 20.0, 30.0) / 10.0 >= 0.4 - 1e-8 for s in starts
    )


def test_preprocessing_stops_on_skipped_window(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from preprocessing import edf_io

    task = SimpleNamespace(windows=[object()])
    monkeypatch.setattr(
        edf_io,
        "process_file_task",
        lambda t: ("train", "bad.edf", np.empty((0, 22, 1000)), None, None, {}),
    )
    with pytest.raises(RuntimeError, match="skipped windows"):
        edf_io.write_dataset({"train": [task], "val": [], "test": []}, tmp_path, 1)
