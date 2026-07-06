"""Metadata preparation for Jazz Trio Database standard classification."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SPLIT_NAMES = ("train", "val", "test")
GROUP_SEPARATOR = " / "


@dataclass(frozen=True)
class PreparedRow:
    """A metadata row with the normalized label used for classification."""

    fname_placeholder: str
    track: str
    standard: str
    year: str
    bandleader: str
    pianist: str
    jtd_300: str

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "PreparedRow":
        return cls(
            fname_placeholder=row["fname_placeholder"],
            track=row["Track"],
            standard=normalize_title(row["Track"]),
            year=row["Year"],
            bandleader=row["Bandleader"],
            pianist=row["Pianist"],
            jtd_300=row["JTD-300"],
        )

    def to_manifest_row(self, label_index: int, split: str) -> dict[str, str | int]:
        return {
            "split": split,
            "label": self.standard,
            "label_index": label_index,
            "fname_placeholder": self.fname_placeholder,
            "track": self.track,
            "year": self.year,
            "bandleader": self.bandleader,
            "pianist": self.pianist,
            "jtd_300": self.jtd_300,
        }

    @property
    def group_id(self) -> str:
        return f"{self.bandleader}{GROUP_SEPARATOR}{self.pianist}"

    def to_grouped_manifest_row(
        self,
        label_index: int,
        group_index: int,
        cv_index: int,
        split: str | None = None,
        fold: int | None = None,
    ) -> dict[str, str | int]:
        row: dict[str, str | int] = {
            "sample_id": self.fname_placeholder,
            "label": self.standard,
            "label_index": label_index,
            "group_id": self.group_id,
            "group_index": group_index,
            "cv_index": cv_index,
            "fname_placeholder": self.fname_placeholder,
            "track": self.track,
            "year": self.year,
            "bandleader": self.bandleader,
            "pianist": self.pianist,
            "jtd_300": self.jtd_300,
        }
        if split is not None:
            row = {"fold": fold if fold is not None else "", "split": split, **row}
        return row


def normalize_title(title: str) -> str:
    """Collapse simple take/version suffixes while preserving readable titles."""

    normalized = title.strip()
    normalized = re.sub(r"\s+alternate take$", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+take\s+\d+$", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+\([^)]*\)$", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def load_rows(metadata_csv: Path) -> list[PreparedRow]:
    with metadata_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [PreparedRow.from_csv_row(row) for row in reader]


def class_counts(rows: Iterable[PreparedRow]) -> Counter[str]:
    return Counter(row.standard for row in rows)


def select_classes(rows: Iterable[PreparedRow], min_performances: int) -> list[dict[str, str | int]]:
    counts = class_counts(rows)
    selected = [
        {"label": label, "count": count}
        for label, count in counts.items()
        if count >= min_performances
    ]
    return sorted(selected, key=lambda item: (-int(item["count"]), str(item["label"])))


def select_group_diverse_classes(
    rows: Iterable[PreparedRow],
    min_groups: int,
) -> list[dict[str, str | int]]:
    by_label: dict[str, list[PreparedRow]] = defaultdict(list)
    for row in rows:
        by_label[row.standard].append(row)

    selected = []
    for label, label_rows in by_label.items():
        groups = {row.group_id for row in label_rows}
        if len(groups) >= min_groups:
            selected.append(
                {
                    "label": label,
                    "count": len(label_rows),
                    "group_count": len(groups),
                }
            )
    return sorted(
        selected,
        key=lambda item: (-int(item["group_count"]), -int(item["count"]), str(item["label"])),
    )


def sample_one_per_label_group(
    rows: Iterable[PreparedRow],
    classes: Iterable[dict[str, str | int]],
    seed: int,
) -> list[PreparedRow]:
    rng = random.Random(seed)
    selected_labels = {str(item["label"]) for item in classes}
    grouped: dict[tuple[str, str], list[PreparedRow]] = defaultdict(list)

    for row in rows:
        if row.standard in selected_labels:
            grouped[(row.standard, row.group_id)].append(row)

    sampled = []
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=lambda row: row.fname_placeholder)
        sampled.append(rng.choice(candidates))

    return sorted(sampled, key=lambda row: (row.standard, row.group_id, row.fname_placeholder))


def assign_group_cv_indices(
    sampled_rows: Iterable[PreparedRow],
    seed: int,
) -> dict[str, dict[str, int]]:
    rng = random.Random(seed)
    groups_by_label: dict[str, list[str]] = defaultdict(list)
    for row in sampled_rows:
        groups_by_label[row.standard].append(row.group_id)

    assignments: dict[str, dict[str, int]] = {}
    for label, group_ids in groups_by_label.items():
        ordered = sorted(set(group_ids))
        rng.shuffle(ordered)
        assignments[label] = {group_id: index for index, group_id in enumerate(ordered)}
    return assignments


def grouped_manifest_fields(include_split: bool = False) -> list[str]:
    fields = [
        "sample_id",
        "label",
        "label_index",
        "group_id",
        "group_index",
        "cv_index",
        "fname_placeholder",
        "track",
        "year",
        "bandleader",
        "pianist",
        "jtd_300",
    ]
    if include_split:
        return ["fold", "split", *fields]
    return fields


def write_grouped_cv_outputs(
    rows: list[PreparedRow],
    classes: list[dict[str, str | int]],
    sampled_rows: list[PreparedRow],
    assignments: dict[str, dict[str, int]],
    output_dir: Path,
    min_groups: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    folds_dir = output_dir / "folds"
    folds_dir.mkdir(exist_ok=True)

    label_to_index = {
        str(item["label"]): index
        for index, item in enumerate(sorted(classes, key=lambda item: str(item["label"])))
    }
    max_folds = max(max(group_map.values()) + 1 for group_map in assignments.values())

    classes_payload = [
        {
            "label": str(item["label"]),
            "label_index": label_to_index[str(item["label"])],
            "metadata_count": int(item["count"]),
            "group_count": int(item["group_count"]),
        }
        for item in sorted(classes, key=lambda item: label_to_index[str(item["label"])])
    ]
    (output_dir / "classes.json").write_text(
        json.dumps(classes_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    selected_rows_path = output_dir / "selected_performances.csv"
    with selected_rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=grouped_manifest_fields())
        writer.writeheader()
        for row in sampled_rows:
            cv_index = assignments[row.standard][row.group_id]
            writer.writerow(
                row.to_grouped_manifest_row(
                    label_index=label_to_index[row.standard],
                    group_index=cv_index,
                    cv_index=cv_index,
                )
            )

    fold_summaries = []
    for fold in range(max_folds):
        fold_dir = folds_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(exist_ok=True)
        test_rows = [
            row for row in sampled_rows if assignments[row.standard][row.group_id] == fold
        ]
        train_rows = [
            row for row in sampled_rows if assignments[row.standard][row.group_id] != fold
        ]
        fold_summaries.append(
            {
                "fold": fold,
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
                "test_classes": len({row.standard for row in test_rows}),
            }
        )

        for split_name, split_rows in (("train", train_rows), ("test", test_rows)):
            with (fold_dir / f"{split_name}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=grouped_manifest_fields(include_split=True))
                writer.writeheader()
                for row in sorted(split_rows, key=lambda item: (item.standard, item.group_id)):
                    cv_index = assignments[row.standard][row.group_id]
                    writer.writerow(
                        row.to_grouped_manifest_row(
                            label_index=label_to_index[row.standard],
                            group_index=cv_index,
                            cv_index=cv_index,
                            split=split_name,
                            fold=fold,
                        )
                    )

    selected_labels = {str(item["label"]) for item in classes}
    summary = {
        "metadata_rows": len(rows),
        "normalized_titles": len(class_counts(rows)),
        "selection": "min_distinct_groups",
        "group_definition": f"Bandleader{GROUP_SEPARATOR}Pianist",
        "min_groups": min_groups,
        "seed": seed,
        "selected_classes": len(classes),
        "selected_metadata_performances": sum(
            1 for row in rows if row.standard in selected_labels
        ),
        "sampled_performances": len(sampled_rows),
        "num_folds": max_folds,
        "folds": fold_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_grouped_cv_metadata(
    metadata_csv: Path,
    output_dir: Path,
    min_groups: int = 4,
    seed: int = 1337,
) -> dict[str, int]:
    rows = load_rows(metadata_csv)
    classes = select_group_diverse_classes(rows, min_groups=min_groups)
    sampled_rows = sample_one_per_label_group(rows, classes, seed=seed)
    assignments = assign_group_cv_indices(sampled_rows, seed=seed)
    write_grouped_cv_outputs(
        rows=rows,
        classes=classes,
        sampled_rows=sampled_rows,
        assignments=assignments,
        output_dir=output_dir,
        min_groups=min_groups,
        seed=seed,
    )
    return {
        "classes": len(classes),
        "sampled_performances": len(sampled_rows),
        "folds": max(max(group_map.values()) + 1 for group_map in assignments.values()),
    }


def split_counts(n_items: int) -> dict[str, int]:
    """Return per-class split counts, preserving val/test for small classes."""

    if n_items < 3:
        raise ValueError("At least three items are required for train/val/test splits.")

    if n_items == 3:
        return {"train": 1, "val": 1, "test": 1}

    val_count = max(1, round(n_items * 0.2))
    test_count = max(1, round(n_items * 0.2))
    train_count = n_items - val_count - test_count

    while train_count < 1:
        if val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break
        train_count = n_items - val_count - test_count

    return {"train": train_count, "val": val_count, "test": test_count}


def make_stratified_split(
    rows: Iterable[PreparedRow],
    classes: Iterable[dict[str, str | int]],
    seed: int,
) -> dict[str, list[PreparedRow]]:
    rng = random.Random(seed)
    selected_labels = {str(item["label"]) for item in classes}
    by_label: dict[str, list[PreparedRow]] = defaultdict(list)

    for row in rows:
        if row.standard in selected_labels:
            by_label[row.standard].append(row)

    splits: dict[str, list[PreparedRow]] = {name: [] for name in SPLIT_NAMES}
    for label in sorted(by_label):
        label_rows = sorted(by_label[label], key=lambda row: row.fname_placeholder)
        rng.shuffle(label_rows)
        counts = split_counts(len(label_rows))

        train_end = counts["train"]
        val_end = train_end + counts["val"]
        splits["train"].extend(label_rows[:train_end])
        splits["val"].extend(label_rows[train_end:val_end])
        splits["test"].extend(label_rows[val_end:])

    for split_rows in splits.values():
        split_rows.sort(key=lambda row: (row.standard, row.fname_placeholder))

    return splits


def write_outputs(
    rows: list[PreparedRow],
    classes: list[dict[str, str | int]],
    splits: dict[str, list[PreparedRow]],
    output_dir: Path,
    min_performances: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    label_to_index = {
        str(item["label"]): index
        for index, item in enumerate(sorted(classes, key=lambda item: str(item["label"])))
    }
    classes_payload = [
        {
            "label": str(item["label"]),
            "label_index": label_to_index[str(item["label"])],
            "count": int(item["count"]),
        }
        for item in sorted(classes, key=lambda item: label_to_index[str(item["label"])])
    ]

    (output_dir / "classes.json").write_text(
        json.dumps(classes_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_fields = [
        "split",
        "label",
        "label_index",
        "fname_placeholder",
        "track",
        "year",
        "bandleader",
        "pianist",
        "jtd_300",
    ]
    for split_name, split_rows in splits.items():
        with (output_dir / f"{split_name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=manifest_fields)
            writer.writeheader()
            for row in split_rows:
                writer.writerow(row.to_manifest_row(label_to_index[row.standard], split_name))

    selected_labels = {str(item["label"]) for item in classes}
    selected_rows = [row for row in rows if row.standard in selected_labels]
    summary = {
        "metadata_rows": len(rows),
        "normalized_titles": len(class_counts(rows)),
        "min_performances": min_performances,
        "seed": seed,
        "selected_classes": len(classes),
        "selected_performances": len(selected_rows),
        "split_sizes": {split: len(split_rows) for split, split_rows in splits.items()},
        "jtd_300_rows": sum(row.jtd_300.lower() == "true" for row in rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_metadata(
    metadata_csv: Path,
    output_dir: Path,
    min_performances: int = 5,
    seed: int = 1337,
) -> dict[str, int]:
    rows = load_rows(metadata_csv)
    classes = select_classes(rows, min_performances)
    splits = make_stratified_split(rows, classes, seed)
    write_outputs(rows, classes, splits, output_dir, min_performances, seed)
    return {
        "classes": len(classes),
        "performances": sum(int(item["count"]) for item in classes),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, default=Path("jtd.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/jtd_standard_24"))
    parser.add_argument("--min-performances", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = prepare_metadata(
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        min_performances=args.min_performances,
        seed=args.seed,
    )
    print(
        "Prepared {classes} classes / {performances} performances "
        "({train} train, {val} val, {test} test).".format(**result)
    )
    return 0
