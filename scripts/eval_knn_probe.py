#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tipofmyear.retrieval.knn_probe import evaluate_knn_probe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument(
        "--classes-json",
        type=Path,
        default=Path("data/processed/jtd_group_cv_16/classes.json"),
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--weights", choices=["distance", "uniform"], default="distance")
    parser.add_argument("--pca-dim", type=int)
    parser.add_argument("--pca-whiten", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = evaluate_knn_probe(
        feature_cache=args.feature_cache,
        fold_dir=args.fold_dir,
        classes_json=args.classes_json,
        results_dir=args.results_dir,
        k=args.k,
        metric=args.metric,
        weights=args.weights,
        pca_dim=args.pca_dim,
        pca_whiten=args.pca_whiten,
        seed=args.seed,
    )
    print(
        "kNN macro acc: {macro_performance_acc:.4f}; "
        "macro top-5: {macro_top5_performance_acc:.4f}".format(**metrics)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
