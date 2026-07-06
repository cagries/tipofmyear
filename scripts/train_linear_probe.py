#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tipofmyear.training.linear_probe import train_linear_probe


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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validation-seed", type=int)
    parser.add_argument(
        "--final-train",
        action="store_true",
        help="Train on the full fold train split without an internal validation split.",
    )
    parser.add_argument("--pca-dim", type=int)
    parser.add_argument("--pca-whiten", action="store_true")
    parser.add_argument("--head", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = train_linear_probe(
        feature_cache=args.feature_cache,
        fold_dir=args.fold_dir,
        classes_json=args.classes_json,
        results_dir=args.results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        validation_seed=args.validation_seed,
        final_train=args.final_train,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        use_wandb=args.wandb_project is not None and not args.no_wandb,
        pca_dim=args.pca_dim,
        pca_whiten=args.pca_whiten,
        head=args.head,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    if args.final_train:
        print(
            "Final test window acc: {final_test_window_acc:.4f}; "
            "final test macro acc: {final_test_macro_performance_acc:.4f}; "
            "final test macro top-5: {final_test_macro_top5_performance_acc:.4f}".format(
                **metrics
            )
        )
    else:
        print(
            "Best val macro top-5: {best_val_macro_top5_performance_acc:.4f}; "
            "final test macro acc: {final_test_macro_performance_acc:.4f}; "
            "final test macro top-5: {final_test_macro_top5_performance_acc:.4f}".format(
                **metrics
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
