"""Classification metrics for grouped-CV experiments."""

from __future__ import annotations

from collections import defaultdict

import torch


def aggregate_window_logits(
    logits: list[torch.Tensor],
    sample_ids: list[str],
    labels: list[int],
    label_names: list[str],
    top_k: int = 5,
) -> list[dict[str, object]]:
    by_sample: dict[str, dict[str, object]] = {}
    for logit, sample_id, label, label_name in zip(logits, sample_ids, labels, label_names):
        if sample_id not in by_sample:
            by_sample[sample_id] = {
                "logits": [],
                "label": label,
                "label_name": label_name,
            }
        by_sample[sample_id]["logits"].append(logit.detach().cpu())

    predictions = []
    for sample_id, payload in by_sample.items():
        mean_logits = torch.stack(payload["logits"]).mean(dim=0)
        k = min(top_k, mean_logits.numel())
        top_values, top_indices = mean_logits.topk(k=k)
        predictions.append(
            {
                "sample_id": sample_id,
                "label": int(payload["label"]),
                "label_name": str(payload["label_name"]),
                "pred": int(mean_logits.argmax().item()),
                "top_indices": [int(index) for index in top_indices.tolist()],
                "top_logits": [float(value) for value in top_values.tolist()],
            }
        )
    return predictions


def macro_accuracy_by_label(predictions: list[dict[str, object]], top_k: int = 1) -> float:
    correct_by_label: dict[int, list[int]] = defaultdict(list)
    for row in predictions:
        label = int(row["label"])
        if top_k == 1:
            correct = label == int(row["pred"])
        else:
            correct = label in [int(index) for index in row.get("top_indices", [])[:top_k]]
        correct_by_label[label].append(correct)
    if not correct_by_label:
        return 0.0
    return sum(sum(vals) / len(vals) for vals in correct_by_label.values()) / len(correct_by_label)


def overall_accuracy(predictions: list[dict[str, object]], top_k: int = 1) -> float:
    if not predictions:
        return 0.0
    if top_k == 1:
        return sum(int(row["label"]) == int(row["pred"]) for row in predictions) / len(predictions)
    return (
        sum(
            int(row["label"]) in [int(index) for index in row.get("top_indices", [])[:top_k]]
            for row in predictions
        )
        / len(predictions)
    )


def window_accuracy(logits: list[torch.Tensor], labels: list[int], top_k: int = 1) -> float:
    if not logits:
        return 0.0
    logit_tensor = torch.stack([logit.detach().cpu() for logit in logits])
    label_tensor = torch.tensor(labels, dtype=torch.long)
    k = min(top_k, logit_tensor.shape[-1])
    top_indices = logit_tensor.topk(k=k, dim=-1).indices
    return float((top_indices == label_tensor[:, None]).any(dim=-1).float().mean().item())
