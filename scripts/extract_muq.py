#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tipofmyear.embeddings.muq import extract_muq_embeddings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/jtd_group_cv_16/manifests/windows_10s_hop5s.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/jtd_group_cv_16/features/muq_10s/"
            "muq_large_msd_iter_layermean_concat.pt"
        ),
    )
    parser.add_argument("--model-id", default="OpenMuQ/MuQ-large-msd-iter")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument(
        "--window-sec",
        type=float,
        help="Window duration in seconds. Defaults to the duration_sec column in the manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = extract_muq_embeddings(
            manifest=args.manifest,
            output=args.output,
            model_id=args.model_id,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            audio_root=args.audio_root,
            window_sec=args.window_sec,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(
        "Extracted {windows} MuQ embeddings with dimension {embedding_dim}: {output}".format(
            **result
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
