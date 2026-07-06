#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tipofmyear.data.augmentations import AudioAugmentationConfig
from tipofmyear.training.hcnn_train import train_hcnn_fold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument(
        "--classes-json",
        type=Path,
        default=Path("data/processed/jtd_group_cv_16/classes.json"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/hcnn"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--window-sec",
        type=float,
        help="Window duration in seconds. Defaults to the duration_sec column in train_windows.csv.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validation-seed", type=int)
    parser.add_argument(
        "--final-train",
        action="store_true",
        help="Train on the full fold train split without an internal validation split.",
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--aug-prob", type=float, default=0.5)
    parser.add_argument("--aug-gain-db", type=float, default=3.0)
    parser.add_argument("--aug-noise-snr-db-min", type=float, default=20.0)
    parser.add_argument("--aug-noise-snr-db-max", type=float, default=40.0)
    parser.add_argument("--aug-crop-jitter-sec", type=float, default=0.5)
    parser.add_argument("--aug-tempo-min", type=float, default=0.97)
    parser.add_argument("--aug-tempo-max", type=float, default=1.03)
    parser.add_argument("--aug-pitch-semitones", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    augmentation_config = AudioAugmentationConfig(
        enabled=args.augment,
        aug_prob=args.aug_prob,
        gain_db=args.aug_gain_db,
        noise_snr_db_min=args.aug_noise_snr_db_min,
        noise_snr_db_max=args.aug_noise_snr_db_max,
        tempo_min=args.aug_tempo_min,
        tempo_max=args.aug_tempo_max,
        pitch_semitones=args.aug_pitch_semitones,
    )
    metrics = train_hcnn_fold(
        fold_dir=args.fold_dir,
        classes_json=args.classes_json,
        results_dir=args.results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        device=args.device,
        window_sec=args.window_sec,
        seed=args.seed,
        validation_seed=args.validation_seed,
        final_train=args.final_train,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        use_wandb=args.wandb_project is not None and not args.no_wandb,
        augmentation_config=augmentation_config,
        crop_jitter_sec=args.aug_crop_jitter_sec,
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
