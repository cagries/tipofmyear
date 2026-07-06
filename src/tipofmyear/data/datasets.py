"""PyTorch datasets for fixed-window audio classification."""

from __future__ import annotations

import csv
from pathlib import Path
import random
from typing import Any, Callable

import soundfile as sf


class WindowedAudioDataset:
    """Load fixed 24 kHz mono waveform windows from a CSV manifest."""

    def __init__(
        self,
        manifest_csv: Path,
        audio_root: Path | None = None,
        target_sample_rate: int = 24000,
        window_sec: float = 10.0,
        pad_short: bool = True,
        transform: Callable[[Any], Any] | None = None,
        crop_jitter_sec: float = 0.0,
    ) -> None:
        self.manifest_csv = manifest_csv.resolve()
        self.audio_root = audio_root.resolve() if audio_root is not None else None
        self.target_sample_rate = target_sample_rate
        self.window_samples = int(round(target_sample_rate * window_sec))
        self.pad_short = pad_short
        self.transform = transform
        self.crop_jitter_samples = int(round(target_sample_rate * crop_jitter_sec))
        with self.manifest_csv.open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))

    def resolve_audio_path(self, audio_path: str) -> Path:
        path = Path(audio_path)
        if path.is_absolute():
            return path

        candidates = []
        if self.audio_root is not None:
            candidates.append(self.audio_root / path)

        candidates.append(Path.cwd() / path)
        candidates.extend(parent / path for parent in self.manifest_csv.parents)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np
        import torch

        row = self.rows[index]
        path = self.resolve_audio_path(row["audio_path"])
        start_sample = int(round(float(row["start_sec"]) * self.target_sample_rate))
        if self.crop_jitter_samples > 0:
            offset = random.randint(-self.crop_jitter_samples, self.crop_jitter_samples)
            start_sample = max(0, start_sample + offset)
        audio, sample_rate = sf.read(
            path,
            start=start_sample,
            frames=self.window_samples,
            dtype="float32",
            always_2d=False,
        )
        if sample_rate != self.target_sample_rate:
            raise ValueError(f"Expected {self.target_sample_rate} Hz, got {sample_rate}: {path}")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) < self.window_samples:
            if not self.pad_short:
                raise ValueError(f"Short window in {path}: {len(audio)} < {self.window_samples}")
            audio = np.pad(audio, (0, self.window_samples - len(audio)))

        waveform = torch.from_numpy(audio.astype("float32")).unsqueeze(0)
        if self.transform is not None:
            waveform = self.transform(waveform)
        return {
            "waveform": waveform,
            "label": torch.tensor(int(row["label_index"]), dtype=torch.long),
            "sample_id": row["sample_id"],
            "window_id": row["window_id"],
            "label_name": row["label"],
            "group_id": row["group_id"],
        }
