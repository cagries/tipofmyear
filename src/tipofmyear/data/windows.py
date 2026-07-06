"""Window manifest generation for fixed-length audio training examples."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_windows_for_performance(
    row: dict[str, str],
    window_sec: float,
    hop_sec: float,
    include_partial: bool = False,
) -> list[dict[str, str]]:
    duration = float(row["duration_sec"])
    if duration < window_sec and not include_partial:
        return []

    if include_partial:
        n_windows = max(1, math.floor(max(0.0, duration - window_sec) / hop_sec) + 1)
    else:
        n_windows = math.floor((duration - window_sec) / hop_sec) + 1

    windows = []
    for index in range(n_windows):
        start_sec = index * hop_sec
        end_sec = start_sec + window_sec
        if end_sec > duration and not include_partial:
            continue
        window_id = f"{row['sample_id']}__{index:05d}"
        windows.append(
            {
                "window_id": window_id,
                "sample_id": row["sample_id"],
                "label": row["label"],
                "label_index": row["label_index"],
                "group_id": row["group_id"],
                "group_index": row["group_index"],
                "cv_index": row["cv_index"],
                "audio_path": row["audio_path"],
                "start_sec": f"{start_sec:.6f}",
                "duration_sec": f"{window_sec:.6f}",
                "sample_rate": row["sample_rate"],
                "year": row["year"],
                "bandleader": row["bandleader"],
                "pianist": row["pianist"],
            }
        )
    return windows


def format_seconds_for_filename(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def all_windows_filename(window_sec: float, hop_sec: float) -> str:
    window = format_seconds_for_filename(window_sec)
    hop = format_seconds_for_filename(hop_sec)
    return f"windows_{window}s_hop{hop}s.csv"


def make_window_manifests(
    performance_manifest: Path,
    folds_dir: Path,
    output_dir: Path,
    window_sec: float = 10.0,
    hop_sec: float = 5.0,
    include_partial: bool = False,
) -> dict[str, int]:
    performances = read_csv(performance_manifest)
    windows_by_sample: dict[str, list[dict[str, str]]] = {}
    all_windows: list[dict[str, str]] = []
    for row in performances:
        windows = build_windows_for_performance(
            row,
            window_sec=window_sec,
            hop_sec=hop_sec,
            include_partial=include_partial,
        )
        windows_by_sample[row["sample_id"]] = windows
        all_windows.extend(windows)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_windows_path = output_dir / all_windows_filename(window_sec, hop_sec)
    write_csv(all_windows_path, all_windows)

    fold_count = 0
    for fold_dir in sorted(path for path in folds_dir.iterdir() if path.is_dir()):
        if not fold_dir.name.startswith("fold_"):
            continue
        fold_out = output_dir / fold_dir.name
        fold_count += 1
        for split in ("train", "test"):
            split_rows = read_csv(fold_dir / f"{split}.csv")
            split_windows: list[dict[str, str]] = []
            for row in split_rows:
                if row["sample_id"] not in windows_by_sample:
                    raise KeyError(
                        f"{row['sample_id']} from {fold_dir / f'{split}.csv'} "
                        f"is missing from {performance_manifest}"
                    )
                for window in windows_by_sample[row["sample_id"]]:
                    split_windows.append(
                        {
                            "fold": row["fold"],
                            "split": split,
                            **window,
                        }
                    )
            write_csv(fold_out / f"{split}_windows.csv", split_windows)

    return {
        "performances": len(performances),
        "windows": len(all_windows),
        "folds": fold_count,
        "all_windows_manifest": str(all_windows_path),
    }
