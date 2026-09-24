"""Build TUSZ v2.0.6 context-aware class-adaptive datasets.

Main rules:
  - csv_bi TERM,seiz defines global seizure event boundaries.
  - csv typed channel annotations assign one merged event class.
  - Drop multi-class conflicts, MYSZ-only/no-typed/generic-only events.
  - Patient-level split is used; official train/dev/eval is not treated as
    the final split.
  - A window is valid if seizure overlap / window length >= 0.40.
  - Val/test use base context-aware windows only.
  - Train uses class-adaptive event-balanced candidate selection:
      CFSZ: base windows; keep full pool by default, sample 8000/epoch in training.
      GNSZ: dense stride 6.0s if base < 4000.
      ABSZ: dense stride 0.2s.
      CTSZ: dense stride 2.0s.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from preprocessing import edf_io

LABEL_MAP = {
    "fnsz": 0,
    "spsz": 0,
    "cpsz": 0,
    "gnsz": 1,
    "absz": 2,
    "tnsz": 3,
    "tcsz": 3,
}
BINARY_POSITIVE_LABELS = set(LABEL_MAP) | {"seiz"}
CLASS_NAMES = {0: "CFSZ", 1: "GNSZ", 2: "ABSZ", 3: "CTSZ"}
SOURCE_SPLITS = ("train", "dev", "eval")


@dataclass(frozen=True)
class CsvEvent:
    channel: str
    start: float
    stop: float
    label: str


@dataclass(frozen=True)
class SeizureEvent:
    event_id: str
    source_split: str
    patient_id: str
    edf_path: str
    recording_duration_sec: float
    event_start: float
    event_stop: float
    class_id: int
    class_name: str
    raw_labels: Tuple[str, ...]


@dataclass
class WindowRecord:
    split: str
    policy: str
    event_id: str
    source_split: str
    patient_id: str
    edf_path: str
    event_start: float
    event_stop: float
    class_id: int
    class_name: str
    start: float
    stop: float
    seizure_overlap_sec: float
    seizure_overlap_ratio: float


def normalize_csv_channel(channel: str) -> str:
    return channel.strip().upper().replace(" ", "")


def overlaps(a_start: float, a_stop: float, b_start: float, b_stop: float) -> bool:
    return max(a_start, b_start) < min(a_stop, b_stop) - 1e-08


def overlap_seconds(
    a_start: float, a_stop: float, b_start: float, b_stop: float
) -> float:
    return max(0.0, min(a_stop, b_stop) - max(a_start, b_start))


def round_time(value: float) -> float:
    return round(float(value), 4)


def read_csv_events(path: Path) -> Tuple[List[CsvEvent], Optional[float]]:
    events: List[CsvEvent] = []
    duration_sec: Optional[float] = None
    if not path.exists():
        return (events, duration_sec)
    with path.open("r", newline="", errors="ignore") as f:
        data_lines = []
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                lower = line.lower()
                if "duration" in lower and "secs" in lower:
                    for token in line.replace("=", " ").split():
                        try:
                            duration_sec = float(token)
                            break
                        except ValueError:
                            continue
                continue
            data_lines.append(raw_line)
        reader = csv.DictReader(data_lines)
        for row in reader:
            try:
                events.append(
                    CsvEvent(
                        channel=normalize_csv_channel(row["channel"]),
                        start=float(row["start_time"]),
                        stop=float(row["stop_time"]),
                        label=row["label"].strip().lower(),
                    )
                )
            except Exception:
                continue
    return (events, duration_sec)


def patient_id_from_edf(input_root: Path, edf_path: Path) -> str:
    rel = edf_path.relative_to(input_root / "edf")
    parts = rel.parts
    return parts[1] if len(parts) >= 2 else "unknown"


def event_label_for_interval(
    typed_events: Sequence[CsvEvent],
    mysz_events: Sequence[CsvEvent],
    generic_events: Sequence[CsvEvent],
    start: float,
    stop: float,
) -> Tuple[Optional[int], str, Tuple[str, ...], Tuple[int, ...]]:
    overlapping_typed = [
        event
        for event in typed_events
        if overlaps(start, stop, event.start, event.stop)
    ]
    raw_labels = tuple(sorted({event.label for event in overlapping_typed}))
    merged_labels = tuple(
        sorted({LABEL_MAP[event.label] for event in overlapping_typed})
    )
    if len(merged_labels) == 1:
        return (merged_labels[0], "keep", raw_labels, merged_labels)
    if len(merged_labels) > 1:
        return (None, "multi_merged_label_conflict", raw_labels, merged_labels)
    if any((overlaps(start, stop, event.start, event.stop) for event in mysz_events)):
        return (None, "mysz_only_or_no_supported_type", tuple(), tuple())
    if any(
        (overlaps(start, stop, event.start, event.stop) for event in generic_events)
    ):
        return (None, "generic_only_no_supported_type", tuple(), tuple())
    return (None, "no_typed_label", tuple(), tuple())


def binary_label_for_window(
    base, channel_events: Sequence[CsvEvent], start: float, stop: float
) -> Tuple[int, ...]:
    positives = [
        event
        for event in channel_events
        if event.label in BINARY_POSITIVE_LABELS
        and overlaps(start, stop, event.start, event.stop)
    ]
    labels = []
    for channel_name in base.CHANNEL_NAMES:
        labels.append(
            int(
                any(
                    (
                        normalize_csv_channel(event.channel) == channel_name
                        for event in positives
                    )
                )
            )
        )
    return tuple(labels)


def discover_events(base, input_root: Path, max_files_per_split: int = 0):
    events: List[SeizureEvent] = []
    channel_cache: Dict[str, List[CsvEvent]] = {}
    stats = Counter()
    dropped_rows = []
    for source_split in SOURCE_SPLITS:
        edf_files = sorted((input_root / "edf" / source_split).rglob("*.edf"))
        if max_files_per_split > 0:
            edf_files = edf_files[:max_files_per_split]
        stats[f"{source_split}_edf_files"] = len(edf_files)
        for edf_path in edf_files:
            (ok, reason) = base.validate_edf_channels(edf_path)
            if not ok:
                stats[f"drop_file_{reason}"] += 1
                continue
            csv_path = edf_path.with_suffix(".csv")
            csv_bi_path = edf_path.with_suffix(".csv_bi")
            (channel_events, duration_sec) = read_csv_events(csv_path)
            (bi_events, bi_duration_sec) = read_csv_events(csv_bi_path)
            channel_cache[str(edf_path)] = channel_events
            typed_events = [
                event for event in channel_events if event.label in LABEL_MAP
            ]
            mysz_events = [event for event in channel_events if event.label == "mysz"]
            generic_events = [
                event for event in channel_events if event.label == "seiz"
            ]
            seizure_events = [
                event
                for event in bi_events
                if event.channel == "TERM" and event.label == "seiz"
            ]
            patient = patient_id_from_edf(input_root, edf_path)
            rec_duration = (
                duration_sec
                or bi_duration_sec
                or max([event.stop for event in channel_events + bi_events] or [0.0])
            )
            stats["csv_bi_seiz_events"] += len(seizure_events)
            for idx, event in enumerate(seizure_events):
                if event.stop <= event.start:
                    stats["drop_zero_duration"] += 1
                    dropped_rows.append(
                        (
                            source_split,
                            str(edf_path),
                            event.start,
                            event.stop,
                            "zero_duration",
                            "",
                            "",
                        )
                    )
                    continue
                (class_id, reason, raw_labels, merged_labels) = (
                    event_label_for_interval(
                        typed_events,
                        mysz_events,
                        generic_events,
                        event.start,
                        event.stop,
                    )
                )
                if reason != "keep" or class_id is None:
                    stats[f"drop_{reason}"] += 1
                    dropped_rows.append(
                        (
                            source_split,
                            str(edf_path),
                            event.start,
                            event.stop,
                            reason,
                            "|".join(raw_labels),
                            "|".join((str(x) for x in merged_labels)),
                        )
                    )
                    continue
                event_id = f"{patient}:{source_split}:{edf_path.stem}:{idx}:{round_time(event.start)}-{round_time(event.stop)}"
                events.append(
                    SeizureEvent(
                        event_id=event_id,
                        source_split=source_split,
                        patient_id=patient,
                        edf_path=str(edf_path),
                        recording_duration_sec=float(rec_duration),
                        event_start=float(event.start),
                        event_stop=float(event.stop),
                        class_id=int(class_id),
                        class_name=CLASS_NAMES[int(class_id)],
                        raw_labels=raw_labels,
                    )
                )
                stats["events_kept"] += 1
                stats[f"events_kept_{CLASS_NAMES[int(class_id)]}"] += 1
    return (events, channel_cache, stats, dropped_rows)


def feasible_start_range(
    ev: SeizureEvent, window_sec: float, ratio: float
) -> Optional[Tuple[float, float]]:
    if ev.recording_duration_sec + 1e-08 < window_sec:
        return None
    min_overlap = ratio * window_sec
    if ev.event_stop - ev.event_start + 1e-08 < min_overlap:
        return None
    lo = max(0.0, ev.event_start + min_overlap - window_sec)
    hi = min(ev.event_stop - min_overlap, ev.recording_duration_sec - window_sec)
    if hi + 1e-08 < lo:
        return None
    return (lo, hi)


def candidate_starts(
    ev: SeizureEvent,
    window_sec: float,
    ratio: float,
    stride_sec: float,
    max_per_event: int,
) -> List[float]:
    feasible = feasible_start_range(ev, window_sec, ratio)
    if feasible is None:
        return []
    (lo, hi) = feasible
    starts = []
    current = lo
    while current <= hi + 1e-08:
        start = round_time(current)
        stop = start + window_sec
        ov = overlap_seconds(start, stop, ev.event_start, ev.event_stop)
        if ov + 1e-08 >= ratio * window_sec:
            starts.append(start)
        current += stride_sec
    if starts and abs(starts[-1] - hi) > 0.0001:
        start = round_time(hi)
        stop = start + window_sec
        ov = overlap_seconds(start, stop, ev.event_start, ev.event_stop)
        if ov + 1e-08 >= ratio * window_sec:
            starts.append(start)
    if not starts:
        starts = [round_time(lo)]
    starts = sorted(set(starts))
    return starts[:max_per_event]


def build_window_record(
    base,
    channel_cache: Dict[str, List[CsvEvent]],
    ev: SeizureEvent,
    split: str,
    policy: str,
    start: float,
    window_sec: float,
) -> Tuple[object, WindowRecord]:
    stop = start + window_sec
    ov = overlap_seconds(start, stop, ev.event_start, ev.event_stop)
    bi_label = binary_label_for_window(base, channel_cache[ev.edf_path], start, stop)
    win = base.WindowSpec(
        start=start, stop=stop, multi_label=ev.class_id, bi_label=bi_label
    )
    win.event_id = ev.event_id
    win.event_start = ev.event_start
    win.event_stop = ev.event_stop
    win.policy = policy
    record = WindowRecord(
        split=split,
        policy=policy,
        event_id=ev.event_id,
        source_split=ev.source_split,
        patient_id=ev.patient_id,
        edf_path=ev.edf_path,
        event_start=round_time(ev.event_start),
        event_stop=round_time(ev.event_stop),
        class_id=ev.class_id,
        class_name=ev.class_name,
        start=round_time(start),
        stop=round_time(stop),
        seizure_overlap_sec=round_time(ov),
        seizure_overlap_ratio=round(ov / window_sec, 4),
    )
    return (win, record)


def read_assignment(path: Path) -> Dict[str, str]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(patient): str(split) for (patient, split) in data.items()}
    assignment = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient = row.get("patient_id") or row.get("patient")
            split = row.get("balanced_split") or row.get("split")
            if patient and split:
                assignment[patient] = split
    return assignment


def event_balanced_select(
    items_by_event: Sequence[Tuple[SeizureEvent, List[float]]],
    target: Optional[int],
    rng: random.Random,
):
    pools = []
    for ev, starts in items_by_event:
        starts = list(starts)
        rng.shuffle(starts)
        if starts:
            pools.append([ev, starts, 0])
    rng.shuffle(pools)
    selected: List[Tuple[SeizureEvent, float]] = []
    while pools and (target is None or len(selected) < target):
        next_pools = []
        for item in pools:
            (ev, starts, idx) = item
            if idx < len(starts) and (target is None or len(selected) < target):
                selected.append((ev, starts[idx]))
                idx += 1
            if idx < len(starts):
                next_pools.append([ev, starts, idx])
        pools = next_pools
    return selected


def stride_policy_name(class_name: str, stride_sec: float) -> str:
    stride_token = f"{stride_sec:g}".replace(".", "p")
    return f"train_dense_{class_name.lower()}_stride{stride_token}"


def build_tasks(base, events, channel_cache, assignment, args):
    rng = random.Random(args.seed)
    window_sec = float(args.duration_sec)
    max_per_event = int(args.max_per_event)
    ratio = float(args.overlap_ratio)
    split_events = {"train": [], "val": [], "test": []}
    for ev in events:
        split = assignment.get(ev.patient_id)
        if split not in split_events:
            raise ValueError(
                f"Missing/invalid split for patient {ev.patient_id}: {split}"
            )
        split_events[split].append(ev)
    tasks_by_split_path = {
        split: defaultdict(list) for split in ("train", "val", "test")
    }
    window_records: List[WindowRecord] = []
    stats = Counter()

    def add_selected(
        split: str, policy: str, selected: Iterable[Tuple[SeizureEvent, float]]
    ):
        for ev, start in selected:
            (win, rec) = build_window_record(
                base, channel_cache, ev, split, policy, start, window_sec
            )
            tasks_by_split_path[split][ev.edf_path].append(win)
            window_records.append(rec)
            stats[f"{split}_windows"] += 1
            stats[f"{split}_class_{ev.class_id}_{ev.class_name}"] += 1
            stats[f"{split}_policy_{policy}"] += 1

    for split in ("val", "test"):
        for ev in split_events[split]:
            starts = candidate_starts(
                ev, window_sec, ratio, window_sec, max_per_event=10000
            )
            add_selected(
                split, "eval_base_context40", [(ev, start) for start in starts]
            )
    train_events_by_class = defaultdict(list)
    for ev in split_events["train"]:
        train_events_by_class[ev.class_id].append(ev)
    base_counts = {}
    base_starts_by_class = {}
    for class_id, evs in train_events_by_class.items():
        grouped = [
            (
                ev,
                candidate_starts(
                    ev, window_sec, ratio, window_sec, max_per_event=10000
                ),
            )
            for ev in evs
        ]
        base_starts_by_class[class_id] = grouped
        base_counts[class_id] = sum((len(starts) for (_, starts) in grouped))
    per_class_target = {
        1: int(args.gnsz_target)
        if args.gnsz_target is not None
        else int(args.min_class_target),
        2: int(args.absz_target)
        if args.absz_target is not None
        else int(args.min_class_target),
        3: int(args.ctsz_target)
        if args.ctsz_target is not None
        else int(args.min_class_target),
    }
    train_policy = {}
    for class_id in range(4):
        base_total = base_counts.get(class_id, 0)
        if class_id == 0:
            epoch_target = min(base_total, args.max_large_class)
            materialized_target = (
                None if base_total > args.max_large_class else epoch_target
            )
            train_policy[class_id] = (
                "train_base_cfsz_pool",
                window_sec,
                materialized_target,
                epoch_target,
            )
        elif class_id == 1:
            target = per_class_target[class_id]
            if base_total >= target:
                epoch_target = target
                materialized_target = target
                train_policy[class_id] = (
                    "train_base_gnsz_pool",
                    window_sec,
                    materialized_target,
                    epoch_target,
                )
            else:
                train_policy[class_id] = (
                    stride_policy_name("GNSZ", args.gnsz_stride_sec),
                    float(args.gnsz_stride_sec),
                    target,
                    target,
                )
        elif class_id == 2:
            target = per_class_target[class_id]
            train_policy[class_id] = (
                stride_policy_name("ABSZ", args.absz_stride_sec),
                float(args.absz_stride_sec),
                target,
                target,
            )
        elif class_id == 3:
            target = per_class_target[class_id]
            train_policy[class_id] = (
                stride_policy_name("CTSZ", args.ctsz_stride_sec),
                float(args.ctsz_stride_sec),
                target,
                target,
            )
    for class_id, evs in sorted(train_events_by_class.items()):
        (policy_name, stride_sec, materialized_target, epoch_target) = train_policy[
            class_id
        ]
        per_event_limit = max_per_event if stride_sec < window_sec else 10000
        grouped = [
            (
                ev,
                candidate_starts(
                    ev, window_sec, ratio, stride_sec, max_per_event=per_event_limit
                ),
            )
            for ev in evs
        ]
        total_candidates = sum((len(starts) for (_, starts) in grouped))
        stats[f"train_candidate_class_{class_id}_{CLASS_NAMES[class_id]}"] = (
            total_candidates
        )
        stats[f"train_base_class_{class_id}_{CLASS_NAMES[class_id]}"] = base_counts.get(
            class_id, 0
        )
        stats[f"train_epoch_target_class_{class_id}_{CLASS_NAMES[class_id]}"] = min(
            int(epoch_target), total_candidates
        )
        selected_target = (
            min(int(materialized_target), total_candidates)
            if materialized_target is not None
            else None
        )
        selected = event_balanced_select(grouped, selected_target, rng)
        add_selected("train", policy_name, selected)
        stats[f"train_selected_class_{class_id}_{CLASS_NAMES[class_id]}"] = len(
            selected
        )
        event_contrib = Counter((ev.event_id for (ev, _) in selected))
        if event_contrib:
            stats[f"train_selected_events_class_{class_id}_{CLASS_NAMES[class_id]}"] = (
                len(event_contrib)
            )
            stats[
                f"train_selected_max_per_event_class_{class_id}_{CLASS_NAMES[class_id]}"
            ] = max(event_contrib.values())
    tasks = {"train": [], "val": [], "test": []}
    for split, by_path in tasks_by_split_path.items():
        for edf_path, windows in sorted(by_path.items()):
            tasks[split].append(
                base.FileTask(split=split, edf_path=edf_path, windows=windows)
            )
    return (tasks, window_records, stats)


def summarize_patients(
    events: Sequence[SeizureEvent], assignment: Dict[str, str]
) -> Dict[str, object]:
    out = {}
    for split in ("train", "val", "test"):
        split_events = [ev for ev in events if assignment.get(ev.patient_id) == split]
        cls = Counter((ev.class_id for ev in split_events))
        patients = {ev.patient_id for ev in split_events}
        out[split] = {
            "patients": len(patients),
            "events": len(split_events),
            "event_class_counts": {str(i): int(cls.get(i, 0)) for i in range(4)},
        }
    return out


def save_window_manifest(path: Path, rows: Sequence[WindowRecord]) -> None:
    fields = (
        list(asdict(rows[0]).keys())
        if rows
        else [
            "split",
            "policy",
            "event_id",
            "source_split",
            "patient_id",
            "edf_path",
            "event_start",
            "event_stop",
            "class_id",
            "class_name",
            "start",
            "stop",
            "seizure_overlap_sec",
            "seizure_overlap_ratio",
        ]
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def window_record_key(
    split: str, row: WindowRecord
) -> Tuple[str, str, int, float, float]:
    return (
        split,
        row.event_id,
        int(row.class_id),
        round_time(row.start),
        round_time(row.stop),
    )


def save_array_order_manifest(path: Path, tasks, rows: Sequence[WindowRecord]) -> None:
    """Save a manifest aligned to split npy row order, with per-split array_index."""
    by_key = defaultdict(deque)
    for row in rows:
        by_key[window_record_key(row.split, row)].append(row)
    fields = ["array_index"] + (
        list(asdict(rows[0]).keys())
        if rows
        else [
            "split",
            "policy",
            "event_id",
            "source_split",
            "patient_id",
            "edf_path",
            "event_start",
            "event_stop",
            "class_id",
            "class_name",
            "start",
            "stop",
            "seizure_overlap_sec",
            "seizure_overlap_ratio",
        ]
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split in ("train", "val", "test"):
            array_index = 0
            for task in tasks[split]:
                for win in task.windows:
                    key = (
                        split,
                        getattr(win, "event_id"),
                        int(win.multi_label),
                        round_time(win.start),
                        round_time(win.stop),
                    )
                    if not by_key[key]:
                        raise RuntimeError(
                            f"Unable to align window manifest row for key={key}"
                        )
                    row = by_key[key].popleft()
                    payload = asdict(row)
                    payload["array_index"] = array_index
                    writer.writerow(payload)
                    array_index += 1
    leftovers = sum((len(values) for values in by_key.values()))
    if leftovers:
        raise RuntimeError(
            f"Array-order manifest alignment left {leftovers} unused rows"
        )


def save_dropped(path: Path, rows) -> None:
    fields = [
        "source_split",
        "edf_path",
        "start",
        "stop",
        "reason",
        "raw_labels",
        "merged_labels",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)


def check_free_space(
    output_dir: Path, expected_windows: int, window_samples: int, min_free_gb: float
) -> None:
    usage_path = output_dir.parent if output_dir.parent.exists() else output_dir
    usage = shutil.disk_usage(usage_path)
    estimated = expected_windows * 22 * window_samples * np.dtype(np.float64).itemsize
    estimated += expected_windows * (22 + 1) * np.dtype(np.int32).itemsize
    projected = usage.free - estimated
    print(
        f"[SPACE] available={usage.free / 1024**3:.2f} GiB estimated_write={estimated / 1024**3:.2f} GiB projected_free={projected / 1024**3:.2f} GiB required_min_free={min_free_gb:.2f} GiB",
        flush=True,
    )
    if projected < min_free_gb * 1024**3:
        raise RuntimeError("Insufficient free space")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-sec", type=float, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--workers", type=int, default=1, choices=[1])
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--max-files-per-split", type=int, default=0)
    parser.add_argument("--overlap-ratio", type=float, default=0.4)
    parser.add_argument("--max-per-event", type=int, default=200)
    parser.add_argument("--gnsz-stride-sec", type=float, default=6.0)
    parser.add_argument("--absz-stride-sec", type=float, default=0.2)
    parser.add_argument("--ctsz-stride-sec", type=float, default=2.0)
    parser.add_argument("--min-class-target", type=int, default=4000)
    parser.add_argument("--gnsz-target", type=int, default=None)
    parser.add_argument("--absz-target", type=int, default=None)
    parser.add_argument("--ctsz-target", type=int, default=None)
    parser.add_argument("--max-large-class", type=int, default=8000)
    parser.add_argument("--min-free-gb", type=float, default=40.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and (not args.overwrite)
        and (not args.metadata_only)
    ):
        raise SystemExit(
            f"Output directory is not empty: {args.output_dir}. Use --overwrite."
        )
    base = edf_io
    base.WINDOW_SEC = float(args.duration_sec)
    base.WINDOW_SAMPLES = int(round(base.TARGET_FS * base.WINDOW_SEC))
    print(f"[discover] input={args.input_root}", flush=True)
    print(
        f"[duration] sec={base.WINDOW_SEC:g} samples={base.WINDOW_SAMPLES}", flush=True
    )
    (events, channel_cache, discover_stats, dropped_rows) = discover_events(
        base, args.input_root, max_files_per_split=args.max_files_per_split
    )
    assignment = read_assignment(args.assignment_csv)
    print(
        f"[assignment] loaded file={args.assignment_csv} patients={len(assignment)}",
        flush=True,
    )
    missing = sorted({ev.patient_id for ev in events} - set(assignment))
    if missing:
        raise RuntimeError(
            f"Assignment missing {len(missing)} patients; first={missing[:10]}"
        )
    (tasks, window_records, build_stats) = build_tasks(
        base, events, channel_cache, assignment, args
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base.write_support_table(args.output_dir / "class_support_table.csv", tasks)
    base.write_stats(
        args.output_dir / "metadata_stats.json", discover_stats + build_stats, tasks
    )
    save_window_manifest(args.output_dir / "window_manifest.csv", window_records)
    save_array_order_manifest(
        args.output_dir / "window_manifest_array_order.csv", tasks, window_records
    )
    save_dropped(args.output_dir / "dropped_events.csv", dropped_rows)
    split_report = summarize_patients(events, assignment)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(args.input_root),
        "output_dir": str(args.output_dir),
        "duration_sec": float(args.duration_sec),
        "overlap_ratio": float(args.overlap_ratio),
        "max_per_event": int(args.max_per_event),
        "gnsz_stride_sec": float(args.gnsz_stride_sec),
        "absz_stride_sec": float(args.absz_stride_sec),
        "ctsz_stride_sec": float(args.ctsz_stride_sec),
        "min_class_target": int(args.min_class_target),
        "class_materialized_targets": {
            "GNSZ": int(args.gnsz_target)
            if args.gnsz_target is not None
            else int(args.min_class_target),
            "ABSZ": int(args.absz_target)
            if args.absz_target is not None
            else int(args.min_class_target),
            "CTSZ": int(args.ctsz_target)
            if args.ctsz_target is not None
            else int(args.min_class_target),
        },
        "max_large_class": int(args.max_large_class),
        "assignment_csv": str(args.assignment_csv) if args.assignment_csv else None,
        "label_map": LABEL_MAP,
        "class_names": CLASS_NAMES,
        "split_event_summary": split_report,
        "discover_stats": {k: int(v) for (k, v) in sorted(discover_stats.items())},
        "build_stats": {k: int(v) for (k, v) in sorted(build_stats.items())},
        "split_support": base.summarize_tasks(tasks),
        "rules": {
            "event_boundary": "csv_bi TERM,seiz",
            "event_label": "csv typed annotations mapped to one final class; conflicts dropped",
            "window_validity": "seizure overlap ratio >= 0.40",
            "val_test": "base context-aware windows only",
            "train": "class-adaptive event-balanced selected candidates",
            "class_policy": {
                "CFSZ": "base context windows; keep full pool when >8000, sample 8000/epoch during training",
                "GNSZ": f"base context windows when base pool reaches the GNSZ target; otherwise stride {args.gnsz_stride_sec:g}s",
                "ABSZ": f"stride {args.absz_stride_sec:g}s",
                "CTSZ": f"stride {args.ctsz_stride_sec:g}s",
            },
        },
    }
    (args.output_dir / "context_adaptive_stats.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "patient_assignment_used.json").write_text(
        json.dumps(assignment, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "[summary]",
        json.dumps(payload["split_support"], indent=2, sort_keys=True),
        flush=True,
    )
    if args.metadata_only:
        return 0
    expected_windows = sum(
        (len(task.windows) for split_tasks in tasks.values() for task in split_tasks)
    )
    check_free_space(
        args.output_dir, expected_windows, base.WINDOW_SAMPLES, args.min_free_gb
    )
    write_stats = base.write_dataset(tasks, args.output_dir, args.workers)
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_audit import audit

    report = audit(args.output_dir)
    (args.output_dir / "data_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    base.write_stats(
        args.output_dir / "preprocess_stats.json",
        discover_stats + build_stats + write_stats,
        tasks,
    )
    print("[done]", args.output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
