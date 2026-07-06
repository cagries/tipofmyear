"""Training loop for the Harmonic CNN baseline."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from tipofmyear.data.augmentations import AudioAugmentationConfig, AudioAugmenter
from tipofmyear.data.datasets import WindowedAudioDataset
from tipofmyear.evaluation.classification import (
    aggregate_window_logits,
    macro_accuracy_by_label,
    overall_accuracy,
    window_accuracy,
)
from tipofmyear.models.harmonic_cnn import HarmonicCNN


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "waveform": torch.stack([item["waveform"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "sample_id": [str(item["sample_id"]) for item in batch],
        "label_name": [str(item["label_name"]) for item in batch],
    }


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def infer_window_sec(manifest_csv: Path) -> float:
    rows = read_csv_rows(manifest_csv)
    if not rows:
        raise ValueError(f"No rows found in manifest: {manifest_csv}")
    try:
        return float(rows[0]["duration_sec"])
    except KeyError as exc:
        raise ValueError(f"Manifest is missing duration_sec: {manifest_csv}") from exc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | list[dict[str, object]]]:
    model.eval()
    logits: list[torch.Tensor] = []
    sample_ids: list[str] = []
    labels: list[int] = []
    label_names: list[str] = []
    with torch.no_grad():
        for batch in loader:
            waveform = batch["waveform"].to(device)
            batch_logits = model(waveform)
            logits.extend(batch_logits.cpu())
            labels.extend(int(label) for label in batch["label"])
            sample_ids.extend(batch["sample_id"])
            label_names.extend(batch["label_name"])

    predictions = aggregate_window_logits(logits, sample_ids, labels, label_names, top_k=5)
    return {
        "window_acc": window_accuracy(logits, labels, top_k=1),
        "performance_acc": overall_accuracy(predictions, top_k=1),
        "macro_performance_acc": macro_accuracy_by_label(predictions, top_k=1),
        "top5_performance_acc": overall_accuracy(predictions, top_k=5),
        "macro_top5_performance_acc": macro_accuracy_by_label(predictions, top_k=5),
        "predictions": predictions,
    }


def metrics_without_predictions(metrics: dict[str, float | list[dict[str, object]]]) -> dict[str, float]:
    return {
        "window_acc": float(metrics["window_acc"]),
        "performance_acc": float(metrics["performance_acc"]),
        "macro_performance_acc": float(metrics["macro_performance_acc"]),
        "top5_performance_acc": float(metrics["top5_performance_acc"]),
        "macro_top5_performance_acc": float(metrics["macro_top5_performance_acc"]),
    }


def split_train_validation_indices(
    rows: list[dict[str, str]],
    seed: int,
) -> tuple[list[int], list[int], list[str]]:
    rng = random.Random(seed)
    samples_by_label: dict[int, list[str]] = {}
    for row in rows:
        label = int(row["label_index"])
        samples_by_label.setdefault(label, [])
        if row["sample_id"] not in samples_by_label[label]:
            samples_by_label[label].append(row["sample_id"])

    val_sample_ids = []
    for label in sorted(samples_by_label):
        sample_ids = sorted(samples_by_label[label])
        if len(sample_ids) < 2:
            raise ValueError(f"Need at least two training performances for label {label}.")
        val_sample_ids.append(rng.choice(sample_ids))

    val_sample_set = set(val_sample_ids)
    train_indices = [index for index, row in enumerate(rows) if row["sample_id"] not in val_sample_set]
    val_indices = [index for index, row in enumerate(rows) if row["sample_id"] in val_sample_set]
    return train_indices, val_indices, sorted(val_sample_ids)


def hcnn_train_validation_indices(
    rows: list[dict[str, str]],
    seed: int,
    final_train: bool = False,
) -> tuple[list[int], list[int], list[str]]:
    if final_train:
        return list(range(len(rows))), [], []
    return split_train_validation_indices(rows, seed=seed)


def write_subset_manifest(path: Path, rows: list[dict[str, str]], indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = [rows[index] for index in indices]
    if not selected:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0].keys()))
        writer.writeheader()
        writer.writerows(selected)


def maybe_init_wandb(
    *,
    enabled: bool,
    project: str | None,
    entity: str | None,
    run_name: str | None,
    mode: str,
    config: dict[str, object],
):
    if not enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb logging was requested, but wandb is not installed.") from exc
    return wandb.init(project=project, entity=entity, name=run_name, mode=mode, config=config)


def train_hcnn_fold(
    fold_dir: Path,
    classes_json: Path,
    results_dir: Path,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    num_workers: int = 2,
    device: str = "auto",
    seed: int = 1337,
    validation_seed: int | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_mode: str = "online",
    use_wandb: bool = False,
    window_sec: float | None = None,
    augmentation_config: AudioAugmentationConfig | None = None,
    crop_jitter_sec: float = 0.0,
    final_train: bool = False,
) -> dict[str, float]:
    random.seed(seed)
    torch.manual_seed(seed)
    validation_seed = seed if validation_seed is None else validation_seed

    classes = json.loads(classes_json.read_text(encoding="utf-8"))
    num_classes = len(classes)
    torch_device = resolve_device(device)

    train_manifest = fold_dir / "train_windows.csv"
    test_manifest = fold_dir / "test_windows.csv"
    resolved_window_sec = window_sec if window_sec is not None else infer_window_sec(train_manifest)
    resolved_augmentation_config = augmentation_config or AudioAugmentationConfig(enabled=False)
    train_transform = (
        AudioAugmenter(resolved_augmentation_config)
        if resolved_augmentation_config.enabled
        else None
    )

    full_train_dataset = WindowedAudioDataset(train_manifest, window_sec=resolved_window_sec)
    augmented_train_dataset = WindowedAudioDataset(
        train_manifest,
        window_sec=resolved_window_sec,
        transform=train_transform,
        crop_jitter_sec=crop_jitter_sec if resolved_augmentation_config.enabled else 0.0,
    )
    test_dataset = WindowedAudioDataset(test_manifest, window_sec=resolved_window_sec)
    train_indices, val_indices, val_sample_ids = hcnn_train_validation_indices(
        full_train_dataset.rows,
        seed=validation_seed,
        final_train=final_train,
    )
    train_dataset = Subset(augmented_train_dataset, train_indices)
    train_eval_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices) if val_indices else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_batch,
        )
        if val_dataset is not None
        else None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )

    model = HarmonicCNN(num_classes=num_classes).to(torch_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    fold_name = fold_dir.name
    run_dir = results_dir / fold_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_subset_manifest(run_dir / "train_internal_windows.csv", full_train_dataset.rows, train_indices)
    if val_indices:
        write_subset_manifest(run_dir / "val_windows.csv", full_train_dataset.rows, val_indices)
    (run_dir / "validation_samples.json").write_text(
        json.dumps(val_sample_ids, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "fold_dir": str(fold_dir),
                "classes_json": str(classes_json),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_workers": num_workers,
                "device": str(torch_device),
                "seed": seed,
                "validation_seed": validation_seed,
                "final_train": final_train,
                "num_classes": num_classes,
                "window_sec": resolved_window_sec,
                "augmentation": {
                    **resolved_augmentation_config.to_dict(),
                    "crop_jitter_sec": crop_jitter_sec,
                },
                "train_windows": len(train_dataset),
                "val_windows": len(val_dataset) if val_dataset is not None else 0,
                "test_windows": len(test_dataset),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    history = []
    best_macro_top5 = -1.0
    wandb_run = maybe_init_wandb(
        enabled=use_wandb and wandb_project is not None,
        project=wandb_project,
        entity=wandb_entity,
        run_name=wandb_run_name,
        mode=wandb_mode,
        config={
            "fold_dir": str(fold_dir),
            "classes_json": str(classes_json),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_workers": num_workers,
            "device": str(torch_device),
            "seed": seed,
            "validation_seed": validation_seed,
            "final_train": final_train,
            "num_classes": num_classes,
            "window_sec": resolved_window_sec,
            "augmentation": {
                **resolved_augmentation_config.to_dict(),
                "crop_jitter_sec": crop_jitter_sec,
            },
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset) if val_dataset is not None else 0,
            "test_windows": len(test_dataset),
        },
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch in tqdm(train_loader):
            waveform = batch["waveform"].to(torch_device)
            labels = batch["label"].to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(waveform)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.numel())
            total_items += int(labels.numel())

        train_metrics = evaluate(model, train_eval_loader, torch_device)
        val_metrics = evaluate(model, val_loader, torch_device) if val_loader is not None else None
        val_macro_top5 = (
            float(val_metrics["macro_top5_performance_acc"]) if val_metrics is not None else 0.0
        )
        if val_metrics is not None and val_macro_top5 > best_macro_top5:
            best_macro_top5 = val_macro_top5
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            (run_dir / "best_val_predictions.json").write_text(
                json.dumps(val_metrics["predictions"], indent=2) + "\n",
                encoding="utf-8",
            )

        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_items),
            "train_window_acc": float(train_metrics["window_acc"]),
            "train_performance_acc": float(train_metrics["performance_acc"]),
            "train_macro_performance_acc": float(train_metrics["macro_performance_acc"]),
        }
        if val_metrics is not None:
            epoch_record.update(
                {
                    "val_window_acc": float(val_metrics["window_acc"]),
                    "val_performance_acc": float(val_metrics["performance_acc"]),
                    "val_macro_performance_acc": float(val_metrics["macro_performance_acc"]),
                    "val_top5_performance_acc": float(val_metrics["top5_performance_acc"]),
                    "val_macro_top5_performance_acc": val_macro_top5,
                }
            )
        history.append(
            epoch_record
        )
        if wandb_run is not None:
            wandb_run.log(epoch_record, step=epoch)

    train_metrics = evaluate(model, train_eval_loader, torch_device)
    val_metrics = evaluate(model, val_loader, torch_device) if val_loader is not None else None
    metrics = evaluate(model, test_loader, torch_device)
    torch.save(model.state_dict(), run_dir / "final_model.pt")
    if final_train:
        torch.save(model.state_dict(), run_dir / "best_model.pt")
    (run_dir / "final_train_predictions.json").write_text(
        json.dumps(train_metrics["predictions"], indent=2) + "\n",
        encoding="utf-8",
    )
    if val_metrics is not None:
        (run_dir / "final_val_predictions.json").write_text(
            json.dumps(val_metrics["predictions"], indent=2) + "\n",
            encoding="utf-8",
        )
    (run_dir / "test_predictions.json").write_text(
        json.dumps(metrics["predictions"], indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "predictions.json").write_text(
        json.dumps(metrics["predictions"], indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_without_predictions(metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    if wandb_run is not None:
        if not final_train:
            wandb_run.summary["best_val_macro_top5_performance_acc"] = best_macro_top5
        wandb_run.summary["final_test_performance_acc"] = float(metrics["performance_acc"])
        wandb_run.summary["final_test_macro_performance_acc"] = float(
            metrics["macro_performance_acc"]
        )
        wandb_run.finish()
    return {
        "best_val_macro_top5_performance_acc": best_macro_top5 if not final_train else 0.0,
        "final_test_window_acc": float(metrics["window_acc"]),
        "final_test_performance_acc": float(metrics["performance_acc"]),
        "final_test_macro_performance_acc": float(metrics["macro_performance_acc"]),
        "final_test_top5_performance_acc": float(metrics["top5_performance_acc"]),
        "final_test_macro_top5_performance_acc": float(metrics["macro_top5_performance_acc"]),
    }
