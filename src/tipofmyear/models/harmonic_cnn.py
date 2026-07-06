"""Minimal Harmonic CNN for single-label standard classification.

Adapted for this project from the public architecture in
minzwon/sota-music-tagging-models. The model keeps the same high-level idea:
learn over a harmonic stack of spectrograms computed from waveform input.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class HarmonicSTFT(nn.Module):
    """Compute a harmonic stack of magnitude STFTs."""

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 513,
        hop_length: int = 256,
        n_harmonics: int = 6,
        semitone_scale: int = 2,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_harmonics = n_harmonics
        self.semitone_scale = semitone_scale
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
        filterbank = self._build_filterbank(freqs)
        self.register_buffer("filterbank", filterbank, persistent=False)

    def _build_filterbank(self, freqs: torch.Tensor) -> torch.Tensor:
        # C1 through C8, with two bins per semitone by default.
        f_min = 32.70319566257483
        f_max = min(self.sample_rate / 2.0, 4186.009044809578)
        bins_per_octave = 12 * self.semitone_scale
        n_bins = int(math.floor(math.log2(f_max / f_min) * bins_per_octave))
        centers = f_min * (2.0 ** (torch.arange(n_bins, dtype=torch.float32) / bins_per_octave))

        banks = []
        for harmonic in range(1, self.n_harmonics + 1):
            harmonic_centers = centers * harmonic
            sigma = harmonic_centers * (2.0 ** (1.0 / bins_per_octave) - 1.0)
            weights = torch.exp(-0.5 * ((freqs[:, None] - harmonic_centers[None, :]) / sigma) ** 2)
            weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-8)
            banks.append(weights)
        return torch.stack(banks, dim=0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(waveform.device),
            return_complex=True,
        ).abs()
        harmonic = torch.einsum("bft,hfq->bhqt", spec, self.filterbank.to(spec.device))
        return torch.log1p(harmonic)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: tuple[int, int]) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HarmonicCNN(nn.Module):
    """Compact Harmonic CNN classifier for fixed-length 24 kHz waveform windows."""

    def __init__(self, num_classes: int, sample_rate: int = 24000) -> None:
        super().__init__()
        self.frontend = HarmonicSTFT(sample_rate=sample_rate)
        self.encoder = nn.Sequential(
            ConvBlock(6, 128, (2, 2)),
            ConvBlock(128, 128, (2, 2)),
            ConvBlock(128, 128, (2, 2)),
            ConvBlock(128, 128, (2, 2)),
            ConvBlock(128, 128, (2, 2)),
            ConvBlock(128, 256, (2, 3)),
            ConvBlock(256, 256, (2, 3)),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.frontend(waveform)
        x = self.encoder(x)
        return self.head(x)
