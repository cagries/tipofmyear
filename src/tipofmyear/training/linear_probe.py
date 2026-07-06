"""Linear probing for cached window embeddings."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tipofmyear.evaluation.classification import (
    aggregate_window_logits,
    macro_accuracy_by_label,
    overall_accuracy,
    window_accuracy,
)
from tipofmyear.training.hcnn_train import (
    hcnn_train_validation_indices,
    maybe_init_wandb,
    metrics_without_predictions,
    read_csv_rows,
    resolve_device,
    write_subset_manifest,
)


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def feature_metadata_for_config(cache: dict[str, Any]) -> dict[str, object]:
    metadata = cache.get("metadata") or {}
    return {
        "model_id": metadata.get("model_id"),
        "pooling": metadata.get("pooling"),
        "sample_rate": metadata.get("sample_rate"),
        "window_sec": metadata.get("window_sec"),
        "manifest": metadata.get("manifest"),
        "num_windows": metadata.get("num_windows"),
        "embedding_dim": metadata.get("embedding_dim"),
    }


class FeatureDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, rows: list[dict[str, str]], indices: list[int]) -> None:
        self.embeddings = embeddings.float()
        self.rows = [rows[index] for index in indices]
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        cache_index = self.indices[index]
        return {
            "embedding": self.embeddings[cache_index],
            "label": torch.tensor(int(row["label_index"]), dtype=torch.long),
            "sample_id": row["sample_id"],
            "window_id": row["window_id"],
            "label_name": row["label"],
        }


def feature_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "embedding": torch.stack([item["embedding"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "sample_id": [str(item["sample_id"]) for item in batch],
        "window_id": [str(item["window_id"]) for item in batch],
        "label_name": [str(item["label_name"]) for item in batch],
    }


def indices_for_manifest_rows(
    cache_window_ids: list[str],
    manifest_rows: list[dict[str, str]],
) -> list[int]:
    index_by_window_id = {window_id: index for index, window_id in enumerate(cache_window_ids)}
    missing = [row["window_id"] for row in manifest_rows if row["window_id"] not in index_by_window_id]
    if missing:
        raise KeyError(f"{len(missing)} window ids are missing from feature cache; first: {missing[0]}")
    return [index_by_window_id[row["window_id"]] for row in manifest_rows]


def apply_pca_if_requested(
    embeddings: torch.Tensor,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
    run_dir: Path,
    pca_dim: int | None,
    pca_whiten: bool,
    seed: int,
) -> tuple[torch.Tensor, list[int], list[int], list[int], dict[str, object]]:
    original_dim = int(embeddings.shape[1])
    if pca_dim is None:
        return (
            embeddings,
            train_indices,
            val_indices,
            test_indices,
            {
                "pca_enabled": False,
                "original_dim": original_dim,
                "effective_dim": original_dim,
            },
        )

    max_components = min(len(train_indices), original_dim)
    if pca_dim > max_components:
        raise ValueError(
            f"pca_dim={pca_dim} exceeds max allowed components {max_components} "
            f"for {len(train_indices)} train windows and original_dim={original_dim}."
        )

    try:
        import joblib
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise RuntimeError("PCA requires scikit-learn and joblib to be installed.") from exc

    pca = PCA(n_components=pca_dim, whiten=pca_whiten, random_state=seed)
    train_features = embeddings[train_indices].cpu().numpy()
    pca.fit(train_features)

    selected_indices = train_indices + val_indices + test_indices
    reduced = pca.transform(embeddings[selected_indices].cpu().numpy())
    reduced_embeddings = torch.from_numpy(reduced).float()
    index_map = {old_index: new_index for new_index, old_index in enumerate(selected_indices)}
    new_train_indices = [index_map[index] for index in train_indices]
    new_val_indices = [index_map[index] for index in val_indices]
    new_test_indices = [index_map[index] for index in test_indices]

    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, run_dir / "pca.joblib")
    explained_variance = float(pca.explained_variance_ratio_.sum())
    metadata = {
        "pca_enabled": True,
        "original_dim": original_dim,
        "effective_dim": pca_dim,
        "pca_dim": pca_dim,
        "pca_whiten": pca_whiten,
        "pca_explained_variance": explained_variance,
    }
    return reduced_embeddings, new_train_indices, new_val_indices, new_test_indices, metadata


def evaluate_probe(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | list[dict[str, object]]]:
    model.eval()
    logits = []
    labels = []
    sample_ids = []
    label_names = []
    with torch.no_grad():
        for batch in loader:
            embeddings = batch["embedding"].to(device)
            batch_logits = model(embeddings).cpu()
            logits.extend(batch_logits)
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


def build_probe_head(
    input_dim: int,
    num_classes: int,
    head: str = "linear",
    hidden_dim: int = 256,
    dropout: float = 0.5,
) -> nn.Module:
    if head == "linear":
        return nn.Linear(input_dim, num_classes)
    if head == "mlp":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    raise ValueError(f"Unsupported probe head {head!r}; expected 'linear' or 'mlp'.")


def train_linear_probe(
    feature_cache: Path,
    fold_dir: Path,
    classes_json: Path,
    results_dir: Path,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    num_workers: int = 0,
    device: str = "auto",
    seed: int = 1337,
    validation_seed: int | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_mode: str = "online",
    use_wandb: bool = False,
    pca_dim: int | None = None,
    pca_whiten: bool = False,
    head: str = "linear",
    hidden_dim: int = 256,
    dropout: float = 0.5,
    final_train: bool = False,
) -> dict[str, float]:
    random.seed(seed)
    torch.manual_seed(seed)
    validation_seed = seed if validation_seed is None else validation_seed
    torch_device = resolve_device(device)

    classes = json.loads(classes_json.read_text(encoding="utf-8"))
    num_classes = len(classes)
    cache = torch_load(feature_cache)
    embeddings = cache["embeddings"].float()
    cache_window_ids = [str(window_id) for window_id in cache["window_ids"]]
    feature_metadata = feature_metadata_for_config(cache)

    train_rows = read_csv_rows(fold_dir / "train_windows.csv")
    test_rows = read_csv_rows(fold_dir / "test_windows.csv")
    train_fold_indices, val_fold_indices, val_sample_ids = hcnn_train_validation_indices(
        train_rows,
        seed=validation_seed,
        final_train=final_train,
    )
    train_rows_internal = [train_rows[index] for index in train_fold_indices]
    val_rows = [train_rows[index] for index in val_fold_indices]

    train_cache_indices = indices_for_manifest_rows(cache_window_ids, train_rows_internal)
    val_cache_indices = indices_for_manifest_rows(cache_window_ids, val_rows) if val_rows else []
    test_cache_indices = indices_for_manifest_rows(cache_window_ids, test_rows)

    run_dir = results_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    embeddings, train_cache_indices, val_cache_indices, test_cache_indices, pca_metadata = (
        apply_pca_if_requested(
            embeddings=embeddings,
            train_indices=train_cache_indices,
            val_indices=val_cache_indices,
            test_indices=test_cache_indices,
            run_dir=run_dir,
            pca_dim=pca_dim,
            pca_whiten=pca_whiten,
            seed=seed,
        )
    )

    train_dataset = FeatureDataset(embeddings, cache["rows"], train_cache_indices)
    val_dataset = FeatureDataset(embeddings, cache["rows"], val_cache_indices) if val_cache_indices else None
    test_dataset = FeatureDataset(embeddings, cache["rows"], test_cache_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=feature_collate,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=feature_collate,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=feature_collate,
        )
        if val_dataset is not None
        else None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=feature_collate,
    )

    model = build_probe_head(
        input_dim=int(embeddings.shape[1]),
        num_classes=num_classes,
        head=head,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(torch_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    write_subset_manifest(run_dir / "train_internal_windows.csv", train_rows, train_fold_indices)
    if val_fold_indices:
        write_subset_manifest(run_dir / "val_windows.csv", train_rows, val_fold_indices)
    (run_dir / "validation_samples.json").write_text(
        json.dumps(val_sample_ids, indent=2) + "\n",
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
            "feature_cache": str(feature_cache),
            "fold_dir": str(fold_dir),
            "classes_json": str(classes_json),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "device": str(torch_device),
            "seed": seed,
            "validation_seed": validation_seed,
            "final_train": final_train,
            "embedding_dim": int(embeddings.shape[1]),
            "num_classes": num_classes,
            "head": head,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "feature_metadata": feature_metadata,
            **pca_metadata,
        },
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "feature_cache": str(feature_cache),
                "fold_dir": str(fold_dir),
                "classes_json": str(classes_json),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "seed": seed,
                "validation_seed": validation_seed,
                "final_train": final_train,
                "num_classes": num_classes,
                "head": head,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "feature_metadata": feature_metadata,
                **pca_metadata,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch in train_loader:
            features = batch["embedding"].to(torch_device)
            labels = batch["label"].to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.numel())
            total_items += int(labels.numel())

        train_metrics = evaluate_probe(model, train_eval_loader, torch_device)
        val_metrics = evaluate_probe(model, val_loader, torch_device) if val_loader is not None else None
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
        history.append(epoch_record)
        if wandb_run is not None:
            wandb_run.log(epoch_record, step=epoch)

    train_metrics = evaluate_probe(model, train_eval_loader, torch_device)
    val_metrics = evaluate_probe(model, val_loader, torch_device) if val_loader is not None else None
    test_metrics = evaluate_probe(model, test_loader, torch_device)
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
        json.dumps(test_metrics["predictions"], indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_without_predictions(test_metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    if wandb_run is not None:
        if not final_train:
            wandb_run.summary["best_val_macro_top5_performance_acc"] = best_macro_top5
        if pca_metadata["pca_enabled"]:
            wandb_run.summary["pca_explained_variance"] = pca_metadata[
                "pca_explained_variance"
            ]
        wandb_run.summary["final_test_macro_performance_acc"] = float(
            test_metrics["macro_performance_acc"]
        )
        wandb_run.summary["final_test_macro_top5_performance_acc"] = float(
            test_metrics["macro_top5_performance_acc"]
        )
        wandb_run.finish()

    return {
        "best_val_macro_top5_performance_acc": best_macro_top5 if not final_train else 0.0,
        "final_test_window_acc": float(test_metrics["window_acc"]),
        "final_test_performance_acc": float(test_metrics["performance_acc"]),
        "final_test_macro_performance_acc": float(test_metrics["macro_performance_acc"]),
        "final_test_top5_performance_acc": float(test_metrics["top5_performance_acc"]),
        "final_test_macro_top5_performance_acc": float(test_metrics["macro_top5_performance_acc"]),
    }
