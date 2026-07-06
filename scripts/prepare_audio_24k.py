#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tipofmyear.data.audio import prepare_audio_24k


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-performances",
        type=Path,
        default=Path("data/processed/jtd_group_cv_16/selected_performances.csv"),
    )
    parser.add_argument("--raw-audio-root", type=Path, default=Path("raw_jtd_wav_path"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/jtd_group_cv_16/audio_24k_mono"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/processed/jtd_group_cv_16/manifests/performances_24k.csv"),
    )
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_audio_24k(
        selected_performances=args.selected_performances,
        raw_audio_root=args.raw_audio_root,
        output_dir=args.output_dir,
        output_manifest=args.output_manifest,
        sample_rate=args.sample_rate,
        overwrite=args.overwrite,
    )
    print(
        "Prepared {count} audio files at {sample_rate} Hz mono; "
        "manifest: {manifest}".format(**result)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
