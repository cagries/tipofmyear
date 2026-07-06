#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count_standards(rows: list[dict[str, str]]) -> list[tuple[str, int]]:
    counts = Counter(row["label"] for row in rows)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def plot_standard_counts(counts: list[tuple[str, int]], output: Path) -> None:
    if not counts:
        raise ValueError("No standard counts to plot.")

    import matplotlib.pyplot as plt

    labels = [label for label, _ in counts]
    values = [count for _, count in counts]
    y_positions = range(len(labels))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig_height = max(4.5, 0.32 * len(labels))
    fig, ax = plt.subplots(figsize=(7.0, fig_height))
    ax.barh(y_positions, values, color="#4C78A8")
    ax.set_yticks(list(y_positions), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of performances")
    ax.set_title("JTD subset performances per standard")
    ax.set_xlim(0, max(values) + 1)

    for y_pos, value in zip(y_positions, values):
        ax.text(value + 0.08, y_pos, str(value), va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/jtd_group_cv_16/manifests/performances_24k.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/jtd_group_cv_16_standard_counts.png"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    counts = count_standards(read_rows(args.manifest))
    plot_standard_counts(counts, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
