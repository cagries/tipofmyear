#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tipofmyear.evaluation.classification import macro_accuracy_by_label, overall_accuracy
from tipofmyear.retrieval.knn_probe import aggregate_window_scores, class_scores_from_neighbors
from tipofmyear.training.hcnn_train import read_csv_rows
from tipofmyear.training.linear_probe import indices_for_manifest_rows, torch_load


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_optional_rows(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(path) if path.exists() else []


def row_maps(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, str]]]]:
    by_window = {row["window_id"]: row for row in rows}
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row)
    return by_window, by_sample


def load_config(result_dir: Path) -> dict[str, Any]:
    config_path = result_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_feature_cache(result_dir: Path, config: dict[str, Any], override: Path | None = None) -> Path:
    if override is not None:
        return override
    projected_cache = result_dir / "projected_features.pt"
    if projected_cache.exists():
        return projected_cache
    try:
        return Path(str(config["feature_cache"]))
    except KeyError as exc:
        raise KeyError("config.json does not contain feature_cache; pass --feature-cache.") from exc


def load_reference_query_rows(
    result_dir: Path,
    config: dict[str, Any],
    query_split: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    fold_dir = Path(str(config["fold_dir"]))
    if query_split == "val":
        reference_rows = read_optional_rows(result_dir / "train_internal_windows.csv")
        query_rows = read_optional_rows(result_dir / "val_windows.csv")
        if not reference_rows:
            reference_rows = read_csv_rows(fold_dir / "train_windows.csv")
        if not query_rows:
            raise FileNotFoundError(
                f"Validation rows not found in {result_dir}. Expected val_windows.csv."
            )
        return reference_rows, query_rows

    if query_split == "test":
        reference_rows = read_optional_rows(result_dir / "train_internal_windows.csv")
        if not reference_rows:
            reference_rows = read_csv_rows(fold_dir / "train_windows.csv")
        query_rows = read_csv_rows(fold_dir / "test_windows.csv")
        return reference_rows, query_rows

    raise ValueError(f"Unsupported query_split {query_split!r}; expected 'val' or 'test'.")


def apply_saved_pca_if_present(
    result_dir: Path,
    embeddings: torch.Tensor,
    selected_indices: list[int],
) -> torch.Tensor:
    pca_path = result_dir / "pca.joblib"
    selected = embeddings[selected_indices].float()
    if not pca_path.exists():
        return selected
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Loading pca.joblib requires joblib.") from exc
    pca = joblib.load(pca_path)
    reduced = pca.transform(selected.cpu().numpy())
    return torch.from_numpy(reduced).float()


def recompute_neighbors(
    feature_cache: Path,
    result_dir: Path,
    reference_rows: list[dict[str, str]],
    query_rows: list[dict[str, str]],
    k: int,
    metric: str,
) -> list[dict[str, object]]:
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise RuntimeError("Recomputing neighbors requires scikit-learn.") from exc

    cache = torch_load(feature_cache)
    embeddings = cache["embeddings"].float()
    cache_window_ids = [str(window_id) for window_id in cache["window_ids"]]
    reference_indices = indices_for_manifest_rows(cache_window_ids, reference_rows)
    query_indices = indices_for_manifest_rows(cache_window_ids, query_rows)

    reference_features = apply_saved_pca_if_present(result_dir, embeddings, reference_indices)
    query_features = apply_saved_pca_if_present(result_dir, embeddings, query_indices)

    if k < 1:
        raise ValueError("k must be >= 1.")
    if k > len(reference_rows):
        raise ValueError(f"k={k} exceeds number of reference rows {len(reference_rows)}.")

    index = NearestNeighbors(n_neighbors=k, metric=metric)
    index.fit(reference_features.cpu().numpy())
    distances, positions = index.kneighbors(query_features.cpu().numpy(), return_distance=True)

    rows = []
    for query_row, row_distances, row_positions in zip(query_rows, distances, positions):
        for rank, (distance, position) in enumerate(zip(row_distances, row_positions), start=1):
            reference_row = reference_rows[int(position)]
            rows.append(
                {
                    "query_window_id": query_row["window_id"],
                    "query_sample_id": query_row["sample_id"],
                    "query_label": query_row["label"],
                    "rank": rank,
                    "neighbor_window_id": reference_row["window_id"],
                    "neighbor_label_index": int(reference_row["label_index"]),
                    "distance": float(distance),
                }
            )
    return rows


def annotate_neighbor_rows(
    neighbor_rows: list[dict[str, object]],
    query_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    query_by_window, _ = row_maps(query_rows)
    reference_by_window, _ = row_maps(reference_rows)
    annotated = []
    for row in neighbor_rows:
        query = query_by_window[str(row["query_window_id"])]
        neighbor = reference_by_window[str(row["neighbor_window_id"])]
        same_group = query["group_id"] == neighbor["group_id"]
        same_sample = query["sample_id"] == neighbor["sample_id"]
        annotated.append(
            {
                **row,
                "query_group_id": query["group_id"],
                "neighbor_group_id": neighbor["group_id"],
                "neighbor_sample_id": neighbor["sample_id"],
                "neighbor_label": neighbor["label"],
                "same_group": same_group,
                "same_sample": same_sample,
            }
        )
    return annotated


def compute_group_overlap_metrics(
    annotated_neighbors: list[dict[str, object]],
    query_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    num_classes: int,
    k: int = 5,
    weights: str = "distance",
) -> tuple[dict[str, float], list[dict[str, object]]]:
    query_by_window, query_by_sample = row_maps(query_rows)
    reference_by_window, _ = row_maps(reference_rows)
    neighbors_by_query: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in annotated_neighbors:
        if int(row["rank"]) <= k:
            neighbors_by_query[str(row["query_window_id"])].append(row)

    total_neighbors = 0
    same_group_neighbors = 0
    same_sample_neighbors = 0
    query_windows_with_same_group = 0
    window_scores = []
    ordered_query_rows = []
    same_group_support_by_sample_class: dict[str, dict[int, bool]] = defaultdict(dict)

    for query_row in query_rows:
        window_id = query_row["window_id"]
        rows = sorted(neighbors_by_query.get(window_id, []), key=lambda item: int(item["rank"]))
        if not rows:
            continue
        total_neighbors += len(rows)
        same_group_count = sum(bool(row["same_group"]) for row in rows)
        same_sample_neighbors += sum(bool(row["same_sample"]) for row in rows)
        same_group_neighbors += same_group_count
        if same_group_count > 0:
            query_windows_with_same_group += 1

        neighbor_labels = [int(row["neighbor_label_index"]) for row in rows]
        distances = [float(row["distance"]) for row in rows]
        window_scores.append(
            class_scores_from_neighbors(
                neighbor_labels=neighbor_labels,
                distances=distances,
                num_classes=num_classes,
                weights=weights,
            )
        )
        ordered_query_rows.append(query_row)

        for row in rows:
            if bool(row["same_group"]):
                same_group_support_by_sample_class[query_row["sample_id"]][
                    int(row["neighbor_label_index"])
                ] = True

    predictions = aggregate_window_scores(window_scores, ordered_query_rows, top_k=5)
    enriched_predictions = []
    top1_supported = []
    top5_supported_counts = []
    top5_any_supported = []
    for prediction in predictions:
        sample_id = str(prediction["sample_id"])
        supported = same_group_support_by_sample_class.get(sample_id, {})
        top_indices = [int(index) for index in prediction["top_indices"][:5]]
        top5_flags = [bool(supported.get(index, False)) for index in top_indices]
        top1_flag = bool(supported.get(int(prediction["pred"]), False))
        top1_supported.append(top1_flag)
        top5_supported_counts.append(sum(top5_flags))
        top5_any_supported.append(any(top5_flags))
        enriched_predictions.append(
            {
                **prediction,
                "top1_same_group_supported": top1_flag,
                "top5_same_group_supported_count": sum(top5_flags),
                "top5_same_group_supported_flags": top5_flags,
            }
        )

    metrics = {
        "same_group_neighbor_rate_at_k": same_group_neighbors / max(1, total_neighbors),
        "same_sample_neighbor_rate_at_k": same_sample_neighbors / max(1, total_neighbors),
        "query_windows_with_any_same_group_neighbor_at_k": query_windows_with_same_group
        / max(1, len(neighbors_by_query)),
        "top1_same_group_supported_rate": sum(top1_supported) / max(1, len(top1_supported)),
        "top5_same_group_supported_mean": sum(top5_supported_counts)
        / max(1, len(top5_supported_counts)),
        "top5_same_group_supported_any_rate": sum(top5_any_supported)
        / max(1, len(top5_any_supported)),
        "window_acc": float(
            sum(
                int(score.argmax().item()) == int(row["label_index"])
                for score, row in zip(window_scores, ordered_query_rows)
            )
            / max(1, len(window_scores))
        ),
        "performance_acc": float(overall_accuracy(predictions, top_k=1)),
        "macro_performance_acc": float(macro_accuracy_by_label(predictions, top_k=1)),
        "top5_performance_acc": float(overall_accuracy(predictions, top_k=5)),
        "macro_top5_performance_acc": float(macro_accuracy_by_label(predictions, top_k=5)),
        "query_windows": float(len(neighbors_by_query)),
        "query_performances": float(len(query_by_sample)),
        "total_neighbors": float(total_neighbors),
    }
    return metrics, enriched_predictions


def load_existing_neighbors(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--query-split", choices=["val", "test"], default="val")
    parser.add_argument("--use-existing-neighbors", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--weights", choices=["distance", "uniform"], default="distance")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.result_dir)
    output_dir = args.output_dir or (args.result_dir / f"group_overlap_{args.query_split}")
    reference_rows, query_rows = load_reference_query_rows(
        result_dir=args.result_dir,
        config=config,
        query_split=args.query_split,
    )

    if args.use_existing_neighbors:
        neighbor_rows = load_existing_neighbors(args.result_dir / "window_neighbors.csv")
    else:
        feature_cache = resolve_feature_cache(args.result_dir, config, override=args.feature_cache)
        neighbor_rows = recompute_neighbors(
            feature_cache=feature_cache,
            result_dir=args.result_dir,
            reference_rows=reference_rows,
            query_rows=query_rows,
            k=args.k,
            metric=args.metric,
        )

    annotated = annotate_neighbor_rows(neighbor_rows, query_rows, reference_rows)
    num_classes = int(config.get("num_classes", 16))
    metrics, predictions = compute_group_overlap_metrics(
        annotated_neighbors=annotated,
        query_rows=query_rows,
        reference_rows=reference_rows,
        num_classes=num_classes,
        k=args.k,
        weights=args.weights,
    )
    payload = {
        "result_dir": str(args.result_dir),
        "query_split": args.query_split,
        "k": args.k,
        "metric": args.metric,
        "weights": args.weights,
        "metrics": metrics,
    }
    write_json(output_dir / "group_overlap_metrics.json", payload)
    write_json(output_dir / "group_overlap_predictions.json", predictions)
    write_csv(output_dir / "group_overlap_neighbors.csv", annotated)
    print(json.dumps(payload["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
