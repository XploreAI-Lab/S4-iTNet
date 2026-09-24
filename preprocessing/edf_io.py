"""Read TUSZ EDF/CSV files and write aligned S4-iTNet arrays.

The output layout is:

    output_dir/
      train.npy
      val.npy
      test.npy
      train_multi_label.npy
      val_multi_label.npy
      test_multi_label.npy
The high-level builder in ``build_dataset.py`` supplies the window duration and
window specifications. This module handles annotations, TCP bipolar conversion,
200 Hz resampling, filtering, and deterministic array writing.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pyedflib
from scipy import signal

TARGET_FS = 200.0
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SEC)
TCP_CHANNELS = [
    ("FP1", "F7", "FP1-F7"),
    ("F7", "T3", "F7-T3"),
    ("T3", "T5", "T3-T5"),
    ("T5", "O1", "T5-O1"),
    ("FP2", "F8", "FP2-F8"),
    ("F8", "T4", "F8-T4"),
    ("T4", "T6", "T4-T6"),
    ("T6", "O2", "T6-O2"),
    ("A1", "T3", "A1-T3"),
    ("T3", "C3", "T3-C3"),
    ("C3", "CZ", "C3-CZ"),
    ("CZ", "C4", "CZ-C4"),
    ("C4", "T4", "C4-T4"),
    ("T4", "A2", "T4-A2"),
    ("FP1", "F3", "FP1-F3"),
    ("F3", "C3", "F3-C3"),
    ("C3", "P3", "C3-P3"),
    ("P3", "O1", "P3-O1"),
    ("FP2", "F4", "FP2-F4"),
    ("F4", "C4", "F4-C4"),
    ("C4", "P4", "C4-P4"),
    ("P4", "O2", "P4-O2"),
]
CHANNEL_NAMES = [name for (_, _, name) in TCP_CHANNELS]
LABEL_MAP = {
    "fnsz": 0,
    "spsz": 0,
    "cpsz": 0,
    "gnsz": 1,
    "absz": 2,
    "tnsz": 3,
    "tcsz": 3,
}
CLASS_NAMES = {0: "CFSZ", 1: "GNSZ", 2: "ABSZ", 3: "CTSZ"}


@dataclass
class CsvEvent:
    channel: str
    start: float
    stop: float
    label: str


@dataclass
class ClassInterval:
    start: float
    stop: float
    multi_label: int


@dataclass
class WindowSpec:
    start: float
    stop: float
    multi_label: int
    bi_label: Tuple[int, ...]


@dataclass
class FileTask:
    split: str
    edf_path: str
    windows: List[WindowSpec]


def normalize_edf_label(label: str) -> str:
    label = label.strip().upper().replace("EEG", "", 1).strip()
    label = label.replace(" ", "")
    return label.split("-")[0]


def normalize_csv_channel(channel: str) -> str:
    return channel.strip().upper().replace(" ", "")


def overlaps(a_start: float, a_stop: float, b_start: float, b_stop: float) -> bool:
    return max(a_start, b_start) < min(a_stop, b_stop) - 1e-08


def read_csv_events(path: Path) -> List[CsvEvent]:
    events: List[CsvEvent] = []
    with path.open("r", newline="", errors="ignore") as f:
        rows = (line for line in f if line.strip() and (not line.startswith("#")))
        reader = csv.DictReader(rows)
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
    return events


def merge_class_intervals(events: Sequence[CsvEvent]) -> List[ClassInterval]:
    by_class: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for event in events:
        if event.label not in LABEL_MAP:
            continue
        if event.stop <= event.start:
            continue
        by_class[LABEL_MAP[event.label]].append((event.start, event.stop))
    merged: List[ClassInterval] = []
    for multi_label, intervals in by_class.items():
        intervals = sorted(intervals)
        if not intervals:
            continue
        (cur_start, cur_stop) = intervals[0]
        for start, stop in intervals[1:]:
            if start <= cur_stop + 1e-08:
                cur_stop = max(cur_stop, stop)
            else:
                merged.append(ClassInterval(cur_start, cur_stop, multi_label))
                (cur_start, cur_stop) = (start, stop)
        merged.append(ClassInterval(cur_start, cur_stop, multi_label))
    return sorted(merged, key=lambda item: (item.start, item.stop, item.multi_label))


def find_edf_files(root: Path, split_dir: str, max_files: int = 0) -> List[Path]:
    files = sorted((root / "edf" / split_dir).rglob("*.edf"))
    if max_files > 0:
        files = files[:max_files]
    return files


def validate_edf_channels(edf_path: Path) -> Tuple[bool, str]:
    try:
        reader = pyedflib.EdfReader(str(edf_path))
        labels = reader.getSignalLabels()
        reader.close()
    except Exception as exc:
        return (False, f"edf_open_error:{type(exc).__name__}")
    available = {normalize_edf_label(label) for label in labels}
    missing = sorted(
        {elec for (a, b, _) in TCP_CHANNELS for elec in (a, b)} - available
    )
    if missing:
        return (False, "missing_channels:" + "|".join(missing))
    return (True, "ok")


def build_windows_for_file(edf_path: Path) -> Tuple[List[WindowSpec], Counter]:
    stats = Counter()
    csv_path = edf_path.with_suffix(".csv")
    csv_bi_path = edf_path.with_suffix(".csv_bi")
    if not csv_path.exists():
        stats["missing_csv"] += 1
        return ([], stats)
    (ok, reason) = validate_edf_channels(edf_path)
    if not ok:
        stats[reason] += 1
        return ([], stats)
    channel_events = read_csv_events(csv_path)
    typed_events = [e for e in channel_events if e.label in LABEL_MAP]
    generic_events = [e for e in channel_events if e.label == "seiz"]
    mysz_events = [e for e in channel_events if e.label == "mysz"]
    if not typed_events:
        if generic_events:
            stats["skip_generic_only_file"] += 1
        elif mysz_events:
            stats["skip_mysz_only_file"] += 1
        else:
            stats["skip_untyped_file"] += 1
        return ([], stats)
    if csv_bi_path.exists():
        stats["has_csv_bi"] += 1
    else:
        stats["missing_csv_bi"] += 1
    class_intervals = merge_class_intervals(typed_events)
    if not class_intervals:
        stats["no_class_intervals"] += 1
        return ([], stats)
    stats["class_intervals"] += len(class_intervals)
    windows: List[WindowSpec] = []
    for interval in class_intervals:
        duration = interval.stop - interval.start
        num_windows = int(math.floor((duration + 1e-08) / WINDOW_SEC))
        if num_windows <= 0:
            stats["skip_short_class_interval"] += 1
            continue
        for idx in range(num_windows):
            start = interval.start + idx * WINDOW_SEC
            stop = start + WINDOW_SEC
            window_typed_events = [
                event
                for event in typed_events
                if overlaps(start, stop, event.start, event.stop)
            ]
            window_labels = {LABEL_MAP[event.label] for event in window_typed_events}
            if any(
                (
                    overlaps(start, stop, event.start, event.stop)
                    for event in mysz_events
                )
            ):
                stats["skip_mysz_window"] += 1
                continue
            if not window_labels:
                if any(
                    (
                        overlaps(start, stop, event.start, event.stop)
                        for event in generic_events
                    )
                ):
                    stats["skip_generic_seiz_window"] += 1
                else:
                    stats["skip_untyped_window"] += 1
                continue
            if interval.multi_label not in window_labels:
                stats["skip_window_label_not_in_interval"] += 1
                continue
            if len(window_labels) > 1:
                stats["skip_multi_target_window"] += 1
                for label in sorted(window_labels):
                    stats[f"skip_multi_target_window_has_{label}"] += 1
                continue
            multi_label = interval.multi_label
            bi = []
            for channel_name in CHANNEL_NAMES:
                value = 0
                for event in window_typed_events:
                    if (
                        normalize_csv_channel(event.channel) == channel_name
                        and event.label in LABEL_MAP
                        and overlaps(start, stop, event.start, event.stop)
                    ):
                        value = 1
                        break
                bi.append(value)
            windows.append(
                WindowSpec(
                    start=start, stop=stop, multi_label=multi_label, bi_label=tuple(bi)
                )
            )
    if windows:
        stats["files_with_windows"] += 1
        stats["windows"] += len(windows)
        for win in windows:
            stats[f"class_{win.multi_label}"] += 1
            stats["bi_positive"] += int(sum(win.bi_label))
    return (windows, stats)


def discover_tasks(
    input_root: Path, max_files_per_split: int = 0
) -> Tuple[Dict[str, List[FileTask]], Counter]:
    split_map = {"train": "train", "dev": "val", "eval": "test"}
    tasks: Dict[str, List[FileTask]] = {out: [] for out in split_map.values()}
    stats = Counter()
    for in_split, out_split in split_map.items():
        edf_files = find_edf_files(input_root, in_split, max_files=max_files_per_split)
        stats[f"{out_split}_edf_files"] = len(edf_files)
        for edf_path in edf_files:
            (windows, file_stats) = build_windows_for_file(edf_path)
            stats.update({f"{out_split}_{k}": v for (k, v) in file_stats.items()})
            if windows:
                tasks[out_split].append(
                    FileTask(split=out_split, edf_path=str(edf_path), windows=windows)
                )
    return (tasks, stats)


def read_signal_resampled(
    reader: pyedflib.EdfReader, index: int, target_len: int
) -> np.ndarray:
    raw = reader.readSignal(index).astype(np.float64, copy=False)
    if raw.size == target_len:
        return raw
    return signal.resample(raw, target_len).astype(np.float64, copy=False)


def build_tcp_recording(edf_path: str) -> Tuple[np.ndarray, float]:
    reader = pyedflib.EdfReader(edf_path)
    try:
        labels = reader.getSignalLabels()
        label_to_index = {}
        for idx, label in enumerate(labels):
            elec = normalize_edf_label(label)
            label_to_index.setdefault(elec, idx)
        duration = float(reader.file_duration)
        target_len = int(round(duration * TARGET_FS))
        electrode_cache: Dict[str, np.ndarray] = {}

        def get_electrode(elec: str) -> np.ndarray:
            if elec not in electrode_cache:
                electrode_cache[elec] = read_signal_resampled(
                    reader, label_to_index[elec], target_len
                )
            return electrode_cache[elec]

        tcp = np.empty((len(TCP_CHANNELS), target_len), dtype=np.float64)
        for ch_idx, (left, right, _) in enumerate(TCP_CHANNELS):
            tcp[ch_idx] = get_electrode(left) - get_electrode(right)
        sos = signal.butter(
            4, [59.0, 61.0], btype="bandstop", fs=TARGET_FS, output="sos"
        )
        if tcp.shape[1] > 30:
            tcp = signal.sosfiltfilt(sos, tcp, axis=1)
        else:
            tcp = signal.sosfilt(sos, tcp, axis=1)
        return (tcp, duration)
    finally:
        reader.close()


def process_file_task(
    task: FileTask,
) -> Tuple[str, str, np.ndarray, np.ndarray, np.ndarray, Counter]:
    stats = Counter()
    try:
        (tcp, _) = build_tcp_recording(task.edf_path)
    except Exception as exc:
        stats[f"process_error:{type(exc).__name__}"] += len(task.windows)
        return (
            task.split,
            task.edf_path,
            np.empty((0, 22, WINDOW_SAMPLES)),
            np.empty((0, 22), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            stats,
        )
    clips = []
    bi_labels = []
    multi_labels = []
    for win in task.windows:
        start_idx = int(round(win.start * TARGET_FS))
        stop_idx = start_idx + WINDOW_SAMPLES
        if start_idx < 0 or stop_idx > tcp.shape[1]:
            stats["skip_window_out_of_bounds"] += 1
            continue
        clip = tcp[:, start_idx:stop_idx]
        if clip.shape != (22, WINDOW_SAMPLES):
            stats["skip_bad_clip_shape"] += 1
            continue
        if not np.isfinite(clip).all():
            stats["skip_nonfinite_clip"] += 1
            continue
        clips.append(clip)
        bi_labels.append(np.asarray(win.bi_label, dtype=np.int32))
        multi_labels.append(np.int32(win.multi_label))
    if not clips:
        return (
            task.split,
            task.edf_path,
            np.empty((0, 22, WINDOW_SAMPLES)),
            np.empty((0, 22), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            stats,
        )
    x = np.stack(clips, axis=0).astype(np.float64, copy=False)
    b = np.stack(bi_labels, axis=0).astype(np.int32, copy=False)
    y = np.asarray(multi_labels, dtype=np.int32)
    stats["written_windows"] += x.shape[0]
    return (task.split, task.edf_path, x, b, y, stats)


def write_dataset(
    tasks: Dict[str, List[FileTask]], output_dir: Path, workers: int
) -> Counter:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    for split in ["train", "val", "test"]:
        split_tasks = tasks[split]
        expected = sum((len(task.windows) for task in split_tasks))
        stats[f"{split}_expected_windows"] = expected
        x_path = output_dir / f"{split}.npy"
        b_path = output_dir / f"{split}_bi_label.npy"
        y_path = output_dir / f"{split}_multi_label.npy"
        x_arr = np.lib.format.open_memmap(
            x_path, mode="w+", dtype=np.float64, shape=(expected, 22, WINDOW_SAMPLES)
        )
        b_arr = np.lib.format.open_memmap(
            b_path, mode="w+", dtype=np.int32, shape=(expected, 22)
        )
        y_arr = np.lib.format.open_memmap(
            y_path, mode="w+", dtype=np.int32, shape=(expected,)
        )
        offset = 0
        len_completed = 0
        iterator: Iterable
        if workers <= 1:
            iterator = (process_file_task(task) for task in split_tasks)
        else:
            pool = ProcessPoolExecutor(max_workers=workers)
            futures = [pool.submit(process_file_task, task) for task in split_tasks]
            iterator = (future.result() for future in futures)
        try:
            for _, edf_path, x, b, y, file_stats in iterator:
                if int(x.shape[0]) != len(split_tasks[len_completed].windows):
                    raise RuntimeError(
                        "EDF processing skipped windows; refusing an unaligned manifest"
                    )
                len_completed += 1
                stats.update({f"{split}_{k}": v for (k, v) in file_stats.items()})
                n = int(x.shape[0])
                if n == 0:
                    continue
                x_arr[offset : offset + n] = x
                b_arr[offset : offset + n] = b
                y_arr[offset : offset + n] = y
                offset += n
                if offset % 500 == 0 or offset == expected:
                    print(f"[{split}] written {offset}/{expected}", flush=True)
        finally:
            if workers > 1:
                pool.shutdown(wait=True)
        x_arr.flush()
        b_arr.flush()
        y_arr.flush()
        if offset != expected:
            print(
                f"[{split}] compacting {offset}/{expected} windows after skips",
                flush=True,
            )
            x_final = np.asarray(x_arr[:offset])
            b_final = np.asarray(b_arr[:offset])
            y_final = np.asarray(y_arr[:offset])
            del x_arr, b_arr, y_arr
            np.save(x_path, x_final)
            np.save(b_path, b_final)
            np.save(y_path, y_final)
        stats[f"{split}_written_windows"] = offset
        if offset > 0:
            y_loaded = np.load(y_path, mmap_mode="r")
            b_loaded = np.load(b_path, mmap_mode="r")
            cls_counts = Counter(map(int, np.asarray(y_loaded)))
            for cls_id, count in cls_counts.items():
                stats[f"{split}_class_{cls_id}_{CLASS_NAMES.get(cls_id, cls_id)}"] = (
                    int(count)
                )
            stats[f"{split}_bi_positive"] = int(np.asarray(b_loaded).sum())
            stats[f"{split}_bi_total"] = int(np.asarray(b_loaded).size)
    return stats


def save_task_manifest(tasks: Dict[str, List[FileTask]], path: Path) -> None:
    payload = {}
    for split, split_tasks in tasks.items():
        payload[split] = [
            {
                "edf_path": task.edf_path,
                "windows": [
                    {
                        "start": win.start,
                        "stop": win.stop,
                        "multi_label": win.multi_label,
                        "class_name": CLASS_NAMES.get(
                            int(win.multi_label), str(win.multi_label)
                        ),
                        "bi_label": list(win.bi_label),
                    }
                    for win in task.windows
                ],
            }
            for task in split_tasks
        ]
    with path.open("w") as f:
        json.dump(payload, f)


def summarize_tasks(tasks: Dict[str, List[FileTask]]) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for split, split_tasks in tasks.items():
        class_counts = Counter()
        windows = 0
        bi_positive = 0
        for task in split_tasks:
            windows += len(task.windows)
            for win in task.windows:
                class_counts[int(win.multi_label)] += 1
                bi_positive += int(sum(win.bi_label))
        summary[split] = {
            "files": len(split_tasks),
            "windows": int(windows),
            "class_counts": {
                str(i): int(class_counts.get(i, 0)) for i in sorted(CLASS_NAMES)
            },
            "class_names": {str(k): v for (k, v) in CLASS_NAMES.items()},
            "bi_positive": int(bi_positive),
            "bi_total": int(windows * len(CHANNEL_NAMES)),
        }
    return summary


def write_support_table(path: Path, tasks: Dict[str, List[FileTask]]) -> None:
    lines = ["split,class_id,class_name,windows"]
    for split in ["train", "val", "test"]:
        counts = Counter()
        for task in tasks.get(split, []):
            counts.update((int(win.multi_label) for win in task.windows))
        for class_id in sorted(CLASS_NAMES):
            lines.append(
                f"{split},{class_id},{CLASS_NAMES[class_id]},{int(counts.get(class_id, 0))}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stats(path: Path, stats: Counter, tasks: Dict[str, List[FileTask]]) -> None:
    payload = {
        "stats": {k: int(v) for (k, v) in sorted(stats.items())},
        "class_names": CLASS_NAMES,
        "label_map": LABEL_MAP,
        "channel_names": CHANNEL_NAMES,
        "target_fs": TARGET_FS,
        "window_sec": WINDOW_SEC,
        "window_samples": WINDOW_SAMPLES,
        "task_files": {split: len(items) for (split, items) in tasks.items()},
        "task_windows": {
            split: sum((len(task.windows) for task in items))
            for (split, items) in tasks.items()
        },
        "split_support": summarize_tasks(tasks),
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
