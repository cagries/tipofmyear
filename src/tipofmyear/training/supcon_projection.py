"""Supervised contrastive projection for cached window embeddings."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from tipofmyear.evaluation.classification import macro_accuracy_by_label, overall_accuracy
from tipofmyear.retrieval.knn_probe import aggregate_window_scores, class_scores_from_neighbors
from tipofmyear.training.hcnn_train import (
    hcnn_train_validation_indices,
    maybe_init_wandb,
    read_csv_rows,
    resolve_device,
    write_subset_manifest,
)
from tipofmyear.training.linear_probe import (
    feature_metadata_for_config,
    indices_for_manifest_rows,
    torch_load,
)


class SupConProjectionModel(nn.Module):
    """Small projection MLP plus a classifier head on projected embeddings."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        projection_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.classifier = nn.Linear(projection_dim, num_classes)

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        projection = F.normalize(self.projector(embeddings), dim=-1)
        return {
            "projection": projection,
            "logits": self.classifier(projection),
        }


SupConProjectionHead = SupConProjectionModel


def supervised_contrastive_loss(
    projections: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: list[str],
    temperature: float = 0.07,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")
    if projections.ndim != 2:
        raise ValueError("projections must be a 2D tensor.")

    device = projections.device
    labels = labels.to(device)
    normalized = F.normalize(projections, dim=-1)
    logits = normalized @ normalized.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    batch_size = labels.numel()
    eye = torch.eye(batch_size, dtype=torch.bool, device=device)
    same_label = labels[:, None] == labels[None, :]
    different_sample = torch.tensor(
        [[left != right for right in sample_ids] for left in sample_ids],
        dtype=torch.bool,
        device=device,
    )
    positive_mask = same_label & different_sample & ~eye
    denominator_mask = ~eye

    log_denominator = torch.logsumexp(logits.masked_fill(~denominator_mask, -torch.inf), dim=1)
    log_prob = logits - log_denominator[:, None]
    positive_counts = positive_mask.sum(dim=1)
    valid = positive_counts > 0
    if not bool(valid.any()):
        return projections.sum() * 0.0
    positive_log_prob = log_prob.masked_fill(~positive_mask, 0.0)
    mean_log_prob = positive_log_prob.sum(dim=1) / positive_counts.clamp_min(1)
    return -mean_log_prob[valid].mean()


def combined_ce_supcon_loss(
    logits: torch.Tensor,
    projections: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: list[str],
    lambda_supcon: float = 0.1,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if lambda_supcon < 0:
        raise ValueError("lambda_supcon must be >= 0.")
    ce_loss = F.cross_entropy(logits, labels)
    supcon_loss = supervised_contrastive_loss(
        projections=projections,
        labels=labels,
        sample_ids=sample_ids,
        temperature=temperature,
    )
    total_loss = ce_loss + lambda_supcon * supcon_loss
    return total_loss, ce_loss, supcon_loss


def build_supcon_batches(
    rows: list[dict[str, str]],
    indices: list[int],
    classes_per_batch: int,
    performances_per_class: int,
    windows_per_performance: int,
    steps: int,
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    by_label_sample: dict[int, dict[str, list[int]]] = {}
    for index in indices:
        row = rows[index]
        label = int(row["label_index"])
        by_label_sample.setdefault(label, {}).setdefault(row["sample_id"], []).append(index)

    eligible_labels = [
        label
        for label, sample_map in by_label_sample.items()
        if len(sample_map) >= performances_per_class
    ]
    if not eligible_labels:
        raise ValueError("No labels have enough performances for supervised contrastive batches.")

    batches = []
    label_count = min(classes_per_batch, len(eligible_labels))
    for _ in range(steps):
        labels = rng.sample(eligible_labels, k=label_count)
        batch = []
        for label in labels:
            sample_map = by_label_sample[label]
            sample_ids = rng.sample(sorted(sample_map), k=performances_per_class)
            for sample_id in sample_ids:
                window_indices = sample_map[sample_id]
                if len(window_indices) >= windows_per_performance:
                    batch.extend(rng.sample(window_indices, k=windows_per_performance))
                else:
                    batch.extend(rng.choice(window_indices) for _ in range(windows_per_performance))
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def project_embeddings(
    model: nn.Module,
    embeddings: torch.Tensor,
    indices: list[int],
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    model.eval()
    projected = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = embeddings[batch_indices].to(device).float()
            output = model(batch)
            if isinstance(output, dict):
                projection = output["projection"]
            else:
                projection = output
            projected.append(projection.cpu())
    if not projected:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(projected, dim=0)


def evaluate_projected_knn(
    reference_features: torch.Tensor,
    reference_rows: list[dict[str, str]],
    query_features: torch.Tensor,
    query_rows: list[dict[str, str]],
    num_classes: int,
    k: int = 5,
    metric: str = "cosine",
    weights: str = "distance",
) -> tuple[dict[str, float], list[dict[str, object]], list[dict[str, object]]]:
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise RuntimeError("SupCon kNN evaluation requires scikit-learn to be installed.") from exc

    if k < 1:
        raise ValueError("k must be >= 1.")
    if k > len(reference_rows):
        raise ValueError(f"k={k} exceeds number of reference windows {len(reference_rows)}.")

    index = NearestNeighbors(n_neighbors=k, metric=metric)
    index.fit(reference_features.cpu().numpy())
    distances, neighbor_positions = index.kneighbors(query_features.cpu().numpy(), return_distance=True)

    reference_labels = [int(row["label_index"]) for row in reference_rows]
    window_scores = []
    neighbor_rows = []
    window_correct = []
    for query_row, row_distances, row_positions in zip(query_rows, distances, neighbor_positions):
        neighbor_labels = [reference_labels[int(position)] for position in row_positions]
        scores = class_scores_from_neighbors(
            neighbor_labels=neighbor_labels,
            distances=[float(distance) for distance in row_distances],
            num_classes=num_classes,
            weights=weights,
        )
        window_scores.append(scores)
        pred = int(scores.argmax().item())
        window_correct.append(pred == int(query_row["label_index"]))

        for rank, (distance, position, label) in enumerate(
            zip(row_distances, row_positions, neighbor_labels),
            start=1,
        ):
            reference_row = reference_rows[int(position)]
            neighbor_rows.append(
                {
                    "query_window_id": query_row["window_id"],
                    "query_sample_id": query_row["sample_id"],
                    "query_label": query_row["label"],
                    "rank": rank,
                    "neighbor_window_id": reference_row["window_id"],
                    "neighbor_label_index": int(label),
                    "distance": float(distance),
                }
            )

    predictions = aggregate_window_scores(window_scores, query_rows, top_k=5)
    metrics = {
        "window_acc": float(sum(window_correct) / max(1, len(window_correct))),
        "performance_acc": float(overall_accuracy(predictions, top_k=1)),
        "macro_performance_acc": float(macro_accuracy_by_label(predictions, top_k=1)),
        "top5_performance_acc": float(overall_accuracy(predictions, top_k=5)),
        "macro_top5_performance_acc": float(macro_accuracy_by_label(predictions, top_k=5)),
    }
    return metrics, predictions, neighbor_rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_neighbor_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No neighbor rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_projected_cache(
    output: Path,
    source_cache: dict[str, Any],
    projected_embeddings: torch.Tensor,
    metadata: dict[str, object],
) -> None:
    cache = {
        "embeddings": projected_embeddings.cpu(),
        "labels": source_cache["labels"],
        "window_ids": source_cache["window_ids"],
        "sample_ids": source_cache["sample_ids"],
        "labels_text": source_cache["labels_text"],
        "group_ids": source_cache["group_ids"],
        "rows": source_cache["rows"],
        "metadata": metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)


def train_supcon_projection(
    feature_cache: Path,
    fold_dir: Path,
    classes_json: Path,
    results_dir: Path,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "auto",
    seed: int = 1337,
    validation_seed: int | None = None,
    hidden_dim: int = 512,
    projection_dim: int = 128,
    dropout: float = 0.1,
    lambda_supcon: float = 0.1,
    temperature: float = 0.07,
    classes_per_batch: int = 8,
    performances_per_class: int = 2,
    windows_per_performance: int = 2,
    steps_per_epoch: int | None = None,
    eval_batch_size: int = 512,
    k: int = 5,
    metric: str = "cosine",
    weights: str = "distance",
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_mode: str = "online",
    use_wandb: bool = False,
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
    feature_metadata = feature_metadata_for_config(cache)
    cache_window_ids = [str(window_id) for window_id in cache["window_ids"]]

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
    fold_train_cache_indices = indices_for_manifest_rows(cache_window_ids, train_rows)

    batch_size = classes_per_batch * performances_per_class * windows_per_performance
    if batch_size < 2:
        raise ValueError("SupCon batch size must be at least 2.")
    resolved_steps_per_epoch = steps_per_epoch or max(1, math.ceil(len(train_cache_indices) / batch_size))

    run_dir = results_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    write_subset_manifest(run_dir / "train_internal_windows.csv", train_rows, train_fold_indices)
    if val_fold_indices:
        write_subset_manifest(run_dir / "val_windows.csv", train_rows, val_fold_indices)
    write_json(run_dir / "validation_samples.json", val_sample_ids)

    model = SupConProjectionModel(
        input_dim=int(embeddings.shape[1]),
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        projection_dim=projection_dim,
        dropout=dropout,
    ).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    config = {
        "feature_cache": str(feature_cache),
        "fold_dir": str(fold_dir),
        "classes_json": str(classes_json),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "device": str(torch_device),
        "seed": seed,
        "validation_seed": validation_seed,
        "final_train": final_train,
        "input_dim": int(embeddings.shape[1]),
        "hidden_dim": hidden_dim,
        "projection_dim": projection_dim,
        "dropout": dropout,
        "objective": "ce_plus_supcon",
        "lambda_supcon": lambda_supcon,
        "temperature": temperature,
        "classifier_head": "linear_on_projection",
        "classes_per_batch": classes_per_batch,
        "performances_per_class": performances_per_class,
        "windows_per_performance": windows_per_performance,
        "batch_size": batch_size,
        "steps_per_epoch": resolved_steps_per_epoch,
        "k": k,
        "metric": metric,
        "weights": weights,
        "num_classes": num_classes,
        "train_windows": len(train_cache_indices),
        "val_windows": len(val_cache_indices),
        "test_windows": len(test_cache_indices),
        "feature_metadata": feature_metadata,
    }
    write_json(run_dir / "config.json", config)

    wandb_run = maybe_init_wandb(
        enabled=use_wandb and wandb_project is not None,
        project=wandb_project,
        entity=wandb_entity,
        run_name=wandb_run_name,
        mode=wandb_mode,
        config=config,
    )

    history = []
    best_val_macro_acc = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        batches = build_supcon_batches(
            rows=cache["rows"],
            indices=train_cache_indices,
            classes_per_batch=classes_per_batch,
            performances_per_class=performances_per_class,
            windows_per_performance=windows_per_performance,
            steps=resolved_steps_per_epoch,
            seed=seed + epoch,
        )
        total_loss = 0.0
        total_ce_loss = 0.0
        total_supcon_loss = 0.0
        for batch_indices in batches:
            batch_embeddings = embeddings[batch_indices].to(torch_device).float()
            batch_rows = [cache["rows"][index] for index in batch_indices]
            labels = torch.tensor(
                [int(row["label_index"]) for row in batch_rows],
                dtype=torch.long,
                device=torch_device,
            )
            sample_ids = [row["sample_id"] for row in batch_rows]
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_embeddings)
            loss, ce_loss, supcon_loss = combined_ce_supcon_loss(
                logits=output["logits"],
                projections=output["projection"],
                labels=labels,
                sample_ids=sample_ids,
                lambda_supcon=lambda_supcon,
                temperature=temperature,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_ce_loss += float(ce_loss.item())
            total_supcon_loss += float(supcon_loss.item())

        train_projected = project_embeddings(
            model,
            embeddings,
            train_cache_indices,
            torch_device,
            batch_size=eval_batch_size,
        )
        val_metrics = None
        val_macro_acc = 0.0
        if val_cache_indices:
            val_projected = project_embeddings(
                model,
                embeddings,
                val_cache_indices,
                torch_device,
                batch_size=eval_batch_size,
            )
            val_metrics, _, _ = evaluate_projected_knn(
                reference_features=train_projected,
                reference_rows=train_rows_internal,
                query_features=val_projected,
                query_rows=val_rows,
                num_classes=num_classes,
                k=k,
                metric=metric,
                weights=weights,
            )
            val_macro_acc = float(val_metrics["macro_performance_acc"])

        if val_metrics is not None and val_macro_acc > best_val_macro_acc:
            best_val_macro_acc = val_macro_acc
            torch.save(model.state_dict(), run_dir / "best_model.pt")

        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(batches)),
            "train_ce_loss": total_ce_loss / max(1, len(batches)),
            "train_supcon_loss": total_supcon_loss / max(1, len(batches)),
        }
        if val_metrics is not None:
            epoch_record.update(
                {
                    "val_window_acc": float(val_metrics["window_acc"]),
                    "val_performance_acc": float(val_metrics["performance_acc"]),
                    "val_macro_performance_acc": val_macro_acc,
                    "val_top5_performance_acc": float(val_metrics["top5_performance_acc"]),
                    "val_macro_top5_performance_acc": float(
                        val_metrics["macro_top5_performance_acc"]
                    ),
                }
            )
        history.append(epoch_record)
        if wandb_run is not None:
            wandb_run.log(epoch_record, step=epoch)

    torch.save(model.state_dict(), run_dir / "final_model.pt")
    if final_train:
        torch.save(model.state_dict(), run_dir / "best_model.pt")
    else:
        best_state = torch_load(run_dir / "best_model.pt")
        model.load_state_dict(best_state)

    all_indices = list(range(len(cache["rows"])))
    projected_all = project_embeddings(
        model,
        embeddings,
        all_indices,
        torch_device,
        batch_size=eval_batch_size,
    )
    projected_metadata = {
        **feature_metadata,
        "projection": "ce_plus_supcon_mlp",
        "objective": "ce_plus_supcon",
        "source_embedding_dim": int(embeddings.shape[1]),
        "embedding_dim": projection_dim,
        "projection_dim": projection_dim,
        "projection_hidden_dim": hidden_dim,
        "projection_dropout": dropout,
        "classifier_head": "linear_on_projection",
        "lambda_supcon": lambda_supcon,
        "temperature": temperature,
        "projection_checkpoint": str(run_dir / "best_model.pt"),
    }
    projected_cache_path = run_dir / "projected_features.pt"
    save_projected_cache(projected_cache_path, cache, projected_all, projected_metadata)

    fold_train_projected = projected_all[fold_train_cache_indices]
    test_projected = projected_all[test_cache_indices]
    test_metrics, test_predictions, neighbor_rows = evaluate_projected_knn(
        reference_features=fold_train_projected,
        reference_rows=train_rows,
        query_features=test_projected,
        query_rows=test_rows,
        num_classes=num_classes,
        k=k,
        metric=metric,
        weights=weights,
    )

    write_json(run_dir / "history.json", history)
    write_json(run_dir / "metrics.json", test_metrics)
    write_json(run_dir / "test_predictions.json", test_predictions)
    write_neighbor_csv(run_dir / "window_neighbors.csv", neighbor_rows)

    if wandb_run is not None:
        if not final_train:
            wandb_run.summary["best_val_macro_performance_acc"] = best_val_macro_acc
        wandb_run.summary["final_test_macro_performance_acc"] = float(
            test_metrics["macro_performance_acc"]
        )
        wandb_run.summary["final_test_macro_top5_performance_acc"] = float(
            test_metrics["macro_top5_performance_acc"]
        )
        wandb_run.finish()

    return {
        "best_val_macro_performance_acc": best_val_macro_acc if not final_train else 0.0,
        "final_test_window_acc": float(test_metrics["window_acc"]),
        "final_test_performance_acc": float(test_metrics["performance_acc"]),
        "final_test_macro_performance_acc": float(test_metrics["macro_performance_acc"]),
        "final_test_top5_performance_acc": float(test_metrics["top5_performance_acc"]),
        "final_test_macro_top5_performance_acc": float(test_metrics["macro_top5_performance_acc"]),
    }
