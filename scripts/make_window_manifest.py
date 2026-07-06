#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tipofmyear.data.windows import make_window_manifests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--performance-manifest",
        type=Path,
        default=Path("data/processed/jtd_group_cv_16/manifests/performances_24k.csv"),
    )
    parser.add_argument(
        "--folds-dir",
        type=Path,
        default=Path("data/processed/jtd_group_cv_16/folds"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/jtd_group_cv_16/manifests"),
    )
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--hop-sec", type=float, default=5.0)
    parser.add_argument("--include-partial", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = make_window_manifests(
        performance_manifest=args.performance_manifest,
        folds_dir=args.folds_dir,
        output_dir=args.output_dir,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        include_partial=args.include_partial,
    )
    print(
        "Prepared {windows} windows from {performances} performances "
        "across {folds} folds: {all_windows_manifest}".format(**result)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
