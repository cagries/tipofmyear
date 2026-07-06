#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tipofmyear.metadata import prepare_grouped_cv_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, default=Path("jtd.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/jtd_group_cv_16"),
    )
    parser.add_argument("--min-groups", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_grouped_cv_metadata(
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        min_groups=args.min_groups,
        seed=args.seed,
    )
    print(
        "Prepared {classes} classes / {sampled_performances} sampled performances "
        "across {folds} leave-one-group-per-standard folds.".format(**result)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
