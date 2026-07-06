"""Frozen MuQ embedding extraction for window manifests."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tipofmyear.data.datasets import WindowedAudioDataset
from tipofmyear.embeddings.mert import mean_pool_hidden_states, mert_collate, read_manifest_rows
from tipofmyear.training.hcnn_train import infer_window_sec, resolve_device


def _get_hidden_states(output: object) -> tuple[torch.Tensor, ...]:
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("MuQ output did not include hidden_states.")
    return tuple(hidden_states)


def extract_muq_embeddings(
    manifest: Path,
    output: Path,
    model_id: str = "OpenMuQ/MuQ-large-msd-iter",
    batch_size: int = 4,
    device: str = "auto",
    num_workers: int = 0,
    audio_root: Path | None = None,
    window_sec: float | None = None,
) -> dict[str, int | str]:
    try:
        from muq import MuQ
    except ImportError as exc:
        raise RuntimeError(
            "MuQ extraction requires the official MuQ package. Install with "
            "`python -m pip install -e '.[muq]'`."
        ) from exc

    torch_device = resolve_device(device)
    model = MuQ.from_pretrained(model_id)
    model = model.to(torch_device).float().eval()

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
            waveforms = batch["waveform"].to(torch_device).float()
            outputs = model(waveforms, output_hidden_states=True)
            embeddings.append(mean_pool_hidden_states(_get_hidden_states(outputs)).cpu())
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
