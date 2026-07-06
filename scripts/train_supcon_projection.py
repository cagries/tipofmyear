#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tipofmyear.training.supcon_projection import train_supcon_projection


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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validation-seed", type=int)
    parser.add_argument(
        "--final-train",
        action="store_true",
        help="Train on the full fold train split without an internal validation split.",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda-supcon", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--classes-per-batch", type=int, default=8)
    parser.add_argument("--performances-per-class", type=int, default=2)
    parser.add_argument("--windows-per-performance", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--weights", choices=["distance", "uniform"], default="distance")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = train_supcon_projection(
        feature_cache=args.feature_cache,
        fold_dir=args.fold_dir,
        classes_json=args.classes_json,
        results_dir=args.results_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        validation_seed=args.validation_seed,
        final_train=args.final_train,
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        lambda_supcon=args.lambda_supcon,
        temperature=args.temperature,
        classes_per_batch=args.classes_per_batch,
        performances_per_class=args.performances_per_class,
        windows_per_performance=args.windows_per_performance,
        steps_per_epoch=args.steps_per_epoch,
        eval_batch_size=args.eval_batch_size,
        k=args.k,
        metric=args.metric,
        weights=args.weights,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        use_wandb=args.wandb_project is not None and not args.no_wandb,
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
            "Best val macro acc: {best_val_macro_performance_acc:.4f}; "
            "final test macro acc: {final_test_macro_performance_acc:.4f}; "
            "final test macro top-5: {final_test_macro_top5_performance_acc:.4f}".format(
                **metrics
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
