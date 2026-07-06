"""Planned frozen-MERT linear-probe experiment entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise SystemExit(
        "MERT linear probing is not implemented yet. "
        f"Expected manifests in {args.manifest_dir} and audio under {args.audio_root}."
    )


if __name__ == "__main__":
    raise SystemExit(main())
