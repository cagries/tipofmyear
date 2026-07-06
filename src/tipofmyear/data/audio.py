"""Audio conversion utilities for the JTD grouped-CV subset."""

from __future__ import annotations

import csv
from pathlib import Path

import soundfile as sf


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_audio_mono(path: Path) -> tuple["object", int]:
    import numpy as np

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    return mono, sample_rate


def resample_audio(audio: "object", orig_sr: int, target_sr: int) -> "object":
    if orig_sr == target_sr:
        return audio

    import torch
    import torchaudio.functional as F

    tensor = torch.as_tensor(audio).unsqueeze(0)
    resampled = F.resample(tensor, orig_freq=orig_sr, new_freq=target_sr)
    return resampled.squeeze(0).cpu().numpy()


def prepare_audio_24k(
    selected_performances: Path,
    raw_audio_root: Path,
    output_dir: Path,
    output_manifest: Path,
    sample_rate: int = 24000,
    overwrite: bool = False,
) -> dict[str, str | int]:
    rows = _read_rows(selected_performances)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, str]] = []
    for row in rows:
        sample_id = row["sample_id"]
        raw_path = raw_audio_root / f"{sample_id}.wav"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw audio for {sample_id}: {raw_path}")

        output_path = output_dir / f"{sample_id}.wav"
        if overwrite or not output_path.exists():
            audio, orig_sr = load_audio_mono(raw_path)
            audio = resample_audio(audio, orig_sr=orig_sr, target_sr=sample_rate)
            sf.write(output_path, audio, sample_rate, subtype="PCM_16")

        info = sf.info(output_path)
        output_rows.append(
            {
                **row,
                "raw_audio_path": str(raw_path),
                "audio_path": str(output_path),
                "sample_rate": str(info.samplerate),
                "num_samples": str(info.frames),
                "duration_sec": f"{info.frames / info.samplerate:.6f}",
            }
        )

    _write_rows(output_manifest, output_rows)
    return {
        "count": len(output_rows),
        "sample_rate": sample_rate,
        "manifest": str(output_manifest),
    }
