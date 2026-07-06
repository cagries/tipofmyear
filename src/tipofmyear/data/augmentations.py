"""Waveform augmentations for small-data audio training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random

import torch


@dataclass(frozen=True)
class AudioAugmentationConfig:
    enabled: bool = False
    aug_prob: float = 0.5
    gain_db: float = 3.0
    noise_snr_db_min: float = 20.0
    noise_snr_db_max: float = 40.0
    tempo_min: float = 0.97
    tempo_max: float = 1.03
    pitch_semitones: float = 0.5

    def to_dict(self) -> dict[str, bool | float]:
        return asdict(self)


def pad_or_trim(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
    if waveform.shape[-1] > target_samples:
        return waveform[..., :target_samples]
    if waveform.shape[-1] < target_samples:
        return torch.nn.functional.pad(waveform, (0, target_samples - waveform.shape[-1]))
    return waveform


def apply_gain(waveform: torch.Tensor, gain_db: float) -> torch.Tensor:
    return waveform * float(10.0 ** (gain_db / 20.0))


def add_noise_at_snr(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
    signal_power = waveform.pow(2).mean().clamp_min(1e-12)
    noise = torch.randn_like(waveform)
    noise_power = noise.pow(2).mean().clamp_min(1e-12)
    target_noise_power = signal_power / float(10.0 ** (snr_db / 10.0))
    return waveform + noise * torch.sqrt(target_noise_power / noise_power)


def time_stretch_fixed(waveform: torch.Tensor, rate: float, sample_rate: int = 24000) -> torch.Tensor:
    if abs(rate - 1.0) < 1e-6:
        return waveform
    target_samples = waveform.shape[-1]
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("Tempo augmentation requires librosa.") from exc
    audio = waveform.squeeze(0).detach().cpu().numpy()
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    return pad_or_trim(torch.as_tensor(stretched, dtype=waveform.dtype).unsqueeze(0), target_samples)


def pitch_shift_fixed(
    waveform: torch.Tensor,
    semitones: float,
    sample_rate: int = 24000,
) -> torch.Tensor:
    if abs(semitones) < 1e-6:
        return waveform
    target_samples = waveform.shape[-1]
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("Pitch augmentation requires librosa.") from exc
    audio = waveform.squeeze(0).detach().cpu().numpy()
    shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=semitones)
    return pad_or_trim(torch.as_tensor(shifted, dtype=waveform.dtype).unsqueeze(0), target_samples)


class AudioAugmenter:
    """Apply independent random waveform augmentations while preserving length."""

    def __init__(self, config: AudioAugmentationConfig, sample_rate: int = 24000) -> None:
        self.config = config
        self.sample_rate = sample_rate

    def _enabled(self) -> bool:
        return self.config.enabled and random.random() < self.config.aug_prob

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.config.enabled:
            return waveform
        target_samples = waveform.shape[-1]
        augmented = waveform

        if self.config.gain_db > 0 and self._enabled():
            gain_db = random.uniform(-self.config.gain_db, self.config.gain_db)
            augmented = apply_gain(augmented, gain_db)

        if self.config.noise_snr_db_max > 0 and self._enabled():
            low = min(self.config.noise_snr_db_min, self.config.noise_snr_db_max)
            high = max(self.config.noise_snr_db_min, self.config.noise_snr_db_max)
            augmented = add_noise_at_snr(augmented, random.uniform(low, high))

        if self.config.tempo_min > 0 and self.config.tempo_max > 0 and self._enabled():
            low = min(self.config.tempo_min, self.config.tempo_max)
            high = max(self.config.tempo_min, self.config.tempo_max)
            augmented = time_stretch_fixed(
                augmented,
                rate=random.uniform(low, high),
                sample_rate=self.sample_rate,
            )

        if self.config.pitch_semitones > 0 and self._enabled():
            semitones = random.uniform(-self.config.pitch_semitones, self.config.pitch_semitones)
            augmented = pitch_shift_fixed(augmented, semitones=semitones, sample_rate=self.sample_rate)

        return pad_or_trim(augmented, target_samples).clamp(-1.0, 1.0)
