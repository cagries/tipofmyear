"""Frozen MERT embedding extraction for window manifests."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from tipofmyear.data.datasets import WindowedAudioDataset
from tipofmyear.training.hcnn_train import infer_window_sec, resolve_device


def mert_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "waveform": torch.stack([item["waveform"].squeeze(0) for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "sample_id": [str(item["sample_id"]) for item in batch],
        "window_id": [str(item["window_id"]) for item in batch],
        "label_name": [str(item["label_name"]) for item in batch],
        "group_id": [str(item["group_id"]) for item in batch],
    }


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_pool_hidden_states(hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
    pooled = [hidden_state.mean(dim=1) for hidden_state in hidden_states]
    return torch.cat(pooled, dim=-1)


def extract_mert_embeddings(
    manifest: Path,
    output: Path,
    model_id: str = "m-a-p/MERT-v1-95M",
    batch_size: int = 4,
    device: str = "auto",
    num_workers: int = 0,
    audio_root: Path | None = None,
    window_sec: float | None = None,
) -> dict[str, int | str]:
    try:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
    except ImportError as exc:
        raise RuntimeError(
            "MERT extraction requires Hugging Face dependencies. Install with "
            "`python -m pip install -e '.[mert]'`."
        ) from exc

    torch_device = resolve_device(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(torch_device)
    model.eval()

    resolved_window_sec = window_sec if window_sec is not None else infer_window_sec(manifest)
    dataset = WindowedAudioDataset(
        manifest,
        audio_root=audio_root,
        target_sample_rate=24000,
        window_sec=resolved_window_sec,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=mert_collate,
    )

    embeddings = []
    labels = []
    window_ids = []
    sample_ids = []
    labels_text = []
    group_ids = []

    with torch.no_grad():
        for batch in loader:
            waveforms = batch["waveform"].numpy()
            inputs = feature_extractor(
                waveforms,
                sampling_rate=24000,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(torch_device) for key, value in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            embeddings.append(mean_pool_hidden_states(outputs.hidden_states).cpu())
            labels.append(batch["label"].cpu())
            window_ids.extend(batch["window_id"])
            sample_ids.extend(batch["sample_id"])
            labels_text.extend(batch["label_name"])
            group_ids.extend(batch["group_id"])

    embedding_tensor = torch.cat(embeddings, dim=0)
    label_tensor = torch.cat(labels, dim=0)
    rows = read_manifest_rows(manifest)
    cache = {
        "embeddings": embedding_tensor,
        "labels": label_tensor,
        "window_ids": window_ids,
        "sample_ids": sample_ids,
        "labels_text": labels_text,
        "group_ids": group_ids,
        "rows": rows,
        "metadata": {
            "model_id": model_id,
            "pooling": "layer_mean_concat",
            "sample_rate": 24000,
            "window_sec": resolved_window_sec,
            "manifest": str(manifest),
            "num_windows": int(embedding_tensor.shape[0]),
            "embedding_dim": int(embedding_tensor.shape[1]),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    return {
        "windows": int(embedding_tensor.shape[0]),
        "embedding_dim": int(embedding_tensor.shape[1]),
        "output": str(output),
    }
