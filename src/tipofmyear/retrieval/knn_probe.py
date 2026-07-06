"""kNN probing over cached window embeddings."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from tipofmyear.evaluation.classification import macro_accuracy_by_label, overall_accuracy
from tipofmyear.training.hcnn_train import read_csv_rows
from tipofmyear.training.linear_probe import (
    apply_pca_if_requested,
    feature_metadata_for_config,
    indices_for_manifest_rows,
    torch_load,
)


def class_scores_from_neighbors(
    neighbor_labels: list[int],
    distances: list[float],
    num_classes: int,
    weights: str = "distance",
) -> torch.Tensor:
    if weights not in {"distance", "uniform"}:
        raise ValueError(f"Unsupported weights {weights!r}; expected 'distance' or 'uniform'.")

    scores = torch.zeros(num_classes, dtype=torch.float32)
    for label, distance in zip(neighbor_labels, distances):
        if weights == "distance":
            weight = 1.0 / (float(distance) + 1e-8)
        else:
            weight = 1.0
        scores[int(label)] += float(weight)
    return scores


def aggregate_window_scores(
    window_scores: list[torch.Tensor],
    rows: list[dict[str, str]],
    top_k: int = 5,
) -> list[dict[str, object]]:
    by_sample: dict[str, dict[str, object]] = {}
    for score, row in zip(window_scores, rows):
        sample_id = row["sample_id"]
        if sample_id not in by_sample:
            by_sample[sample_id] = {
                "scores": [],
                "label": int(row["label_index"]),
                "label_name": row["label"],
            }
        by_sample[sample_id]["scores"].append(score)

    predictions = []
    for sample_id, payload in by_sample.items():
        mean_scores = torch.stack(payload["scores"]).mean(dim=0)
        k = min(top_k, mean_scores.numel())
        top_values, top_indices = mean_scores.topk(k=k)
        predictions.append(
            {
                "sample_id": sample_id,
                "label": int(payload["label"]),
                "label_name": str(payload["label_name"]),
                "pred": int(top_indices[0].item()),
                "top_indices": [int(index) for index in top_indices.tolist()],
                "top_logits": [float(value) for value in top_values.tolist()],
            }
        )
    return predictions


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


def evaluate_knn_probe(
    feature_cache: Path,
    fold_dir: Path,
    classes_json: Path,
    results_dir: Path,
    k: int = 5,
    metric: str = "cosine",
    weights: str = "distance",
    pca_dim: int | None = None,
    pca_whiten: bool = False,
    seed: int = 1337,
) -> dict[str, float]:
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise RuntimeError("kNN probing requires scikit-learn to be installed.") from exc

    classes = json.loads(classes_json.read_text(encoding="utf-8"))
    num_classes = len(classes)
    cache = torch_load(feature_cache)
    embeddings = cache["embeddings"].float()
    cache_window_ids = [str(window_id) for window_id in cache["window_ids"]]
    feature_metadata = feature_metadata_for_config(cache)
    train_rows = read_csv_rows(fold_dir / "train_windows.csv")
    test_rows = read_csv_rows(fold_dir / "test_windows.csv")
    train_indices = indices_for_manifest_rows(cache_window_ids, train_rows)
    test_indices = indices_for_manifest_rows(cache_window_ids, test_rows)

    results_dir.mkdir(parents=True, exist_ok=True)
    embeddings, train_indices, _, test_indices, pca_metadata = apply_pca_if_requested(
        embeddings=embeddings,
        train_indices=train_indices,
        val_indices=[],
        test_indices=test_indices,
        run_dir=results_dir,
        pca_dim=pca_dim,
        pca_whiten=pca_whiten,
        seed=seed,
    )

    if k < 1:
        raise ValueError("k must be >= 1.")
    if k > len(train_indices):
        raise ValueError(f"k={k} exceeds number of train windows {len(train_indices)}.")

    train_features = embeddings[train_indices].cpu().numpy()
    test_features = embeddings[test_indices].cpu().numpy()
    train_labels = [int(cache["rows"][index]["label_index"]) for index in train_indices]
    train_window_ids = [str(cache["rows"][index]["window_id"]) for index in train_indices]

    index = NearestNeighbors(n_neighbors=k, metric=metric)
    index.fit(train_features)
    distances, neighbor_positions = index.kneighbors(test_features, return_distance=True)

    window_scores = []
    neighbor_rows = []
    window_correct = []
    for test_pos, (row, row_distances, row_positions) in enumerate(
        zip(test_rows, distances, neighbor_positions)
    ):
        neighbor_labels = [train_labels[int(position)] for position in row_positions]
        scores = class_scores_from_neighbors(
            neighbor_labels=neighbor_labels,
            distances=[float(distance) for distance in row_distances],
            num_classes=num_classes,
            weights=weights,
        )
        window_scores.append(scores)
        pred = int(scores.argmax().item())
        window_correct.append(pred == int(row["label_index"]))

        for rank, (distance, position, label) in enumerate(
            zip(row_distances, row_positions, neighbor_labels),
            start=1,
        ):
            neighbor_rows.append(
                {
                    "query_window_id": row["window_id"],
                    "query_sample_id": row["sample_id"],
                    "query_label": row["label"],
                    "rank": rank,
                    "neighbor_window_id": train_window_ids[int(position)],
                    "neighbor_label_index": int(label),
                    "distance": float(distance),
                }
            )

    predictions = aggregate_window_scores(window_scores, test_rows, top_k=5)
    metrics = {
        "window_acc": float(sum(window_correct) / max(1, len(window_correct))),
        "performance_acc": float(overall_accuracy(predictions, top_k=1)),
        "macro_performance_acc": float(macro_accuracy_by_label(predictions, top_k=1)),
        "top5_performance_acc": float(overall_accuracy(predictions, top_k=5)),
        "macro_top5_performance_acc": float(macro_accuracy_by_label(predictions, top_k=5)),
    }
    config = {
        "feature_cache": str(feature_cache),
        "fold_dir": str(fold_dir),
        "classes_json": str(classes_json),
        "k": k,
        "metric": metric,
        "weights": weights,
        "num_classes": num_classes,
        "train_windows": len(train_indices),
        "test_windows": len(test_indices),
        "feature_metadata": feature_metadata,
        **pca_metadata,
    }

    write_json(results_dir / "config.json", config)
    write_json(results_dir / "metrics.json", metrics)
    write_json(results_dir / "test_predictions.json", predictions)
    write_neighbor_csv(results_dir / "window_neighbors.csv", neighbor_rows)
    return metrics
