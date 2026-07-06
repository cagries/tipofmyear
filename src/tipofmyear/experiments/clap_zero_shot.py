"""Planned CLAP zero-shot standard-classification entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument(
        "--prompt-template",
        default="a jazz trio performance of {label}",
        help="Prompt template used to render labels from classes.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise SystemExit(
        "CLAP zero-shot scoring is not implemented yet. "
        f"Expected manifests in {args.manifest_dir}, audio under {args.audio_root}, "
        f"and prompt template {args.prompt_template!r}."
    )


if __name__ == "__main__":
    raise SystemExit(main())
