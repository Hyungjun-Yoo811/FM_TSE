from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


def _hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _create_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> torch.Tensor:
    f_max = min(f_max, sample_rate / 2.0)
    fft_freqs = torch.linspace(0.0, sample_rate / 2.0, steps=(n_fft // 2) + 1)

    mel_points = torch.linspace(
        _hz_to_mel(torch.tensor(f_min)),
        _hz_to_mel(torch.tensor(f_max)),
        steps=n_mels + 2,
    )
    hz_points = _mel_to_hz(mel_points)

    filterbank = torch.zeros(n_mels, (n_fft // 2) + 1)
    for mel_idx in range(n_mels):
        left = hz_points[mel_idx]
        center = hz_points[mel_idx + 1]
        right = hz_points[mel_idx + 2]

        up_slope = (fft_freqs - left) / max((center - left).item(), 1e-8)
        down_slope = (right - fft_freqs) / max((right - center).item(), 1e-8)
        filterbank[mel_idx] = torch.clamp(torch.minimum(up_slope, down_slope), min=0.0)

    enorm = 2.0 / torch.clamp(hz_points[2:n_mels + 2] - hz_points[:n_mels], min=1e-8)
    filterbank = filterbank * enorm.unsqueeze(1)
    return filterbank


@dataclass
class MelSpectrogramConfig:
    sample_rate: int
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float | None = None
    log_epsilon: float = 1e-5


class MelSpectrogramTransform(nn.Module):
    def __init__(self, config: MelSpectrogramConfig) -> None:
        super().__init__()
        self.config = config
        f_max = config.f_max if config.f_max is not None else config.sample_rate / 2.0

        window = torch.hann_window(config.win_length)
        mel_basis = _create_mel_filterbank(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=f_max,
        )
        mel_basis_pinv = torch.linalg.pinv(mel_basis)
        self.register_buffer("window", window, persistent=False)
        self.register_buffer("mel_basis", mel_basis, persistent=False)
        self.register_buffer("mel_basis_pinv", mel_basis_pinv, persistent=False)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 2:
            raise ValueError(f"Expected audio with shape [batch, time], got {tuple(audio.shape)}")

        stft = torch.stft(
            audio,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window,
            return_complex=True,
        )
        magnitude = stft.abs()
        mel = torch.matmul(self.mel_basis.unsqueeze(0), magnitude)
        return torch.log(mel + self.config.log_epsilon)

    def inverse(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.ndim != 3:
            raise ValueError(f"Expected mel with shape [batch, n_mels, frames], got {tuple(mel.shape)}")
        mel_linear = torch.exp(mel).clamp_min(self.config.log_epsilon)
        magnitude = torch.matmul(self.mel_basis_pinv.unsqueeze(0), mel_linear).clamp_min(0.0)
        return magnitude

    def griffin_lim(
        self,
        mel: torch.Tensor,
        num_iters: int = 32,
        length: int | None = None,
    ) -> torch.Tensor:
        magnitude = self.inverse(mel)
        phase = 2.0 * math.pi * torch.rand_like(magnitude)
        complex_spec = torch.polar(magnitude, phase)

        for _ in range(num_iters):
            audio = torch.istft(
                complex_spec,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                window=self.window,
                length=length,
            )
            rebuilt = torch.stft(
                audio,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                window=self.window,
                return_complex=True,
            )
            complex_spec = magnitude * torch.exp(1j * torch.angle(rebuilt))

        return torch.istft(
            complex_spec,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window,
            length=length,
        )

    @property
    def n_mels(self) -> int:
        return self.config.n_mels

    def frame_count(self, signal_length: int) -> int:
        return 1 + signal_length // self.config.hop_length
