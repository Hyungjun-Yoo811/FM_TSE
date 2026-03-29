from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnrollmentEncoder1D(nn.Module):
    def __init__(self, speaker_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.GELU(),
            nn.Conv1d(128, speaker_dim, kernel_size=15, stride=2, padding=7),
            nn.GELU(),
        )

    def forward(self, enrollment: torch.Tensor) -> torch.Tensor:
        hidden = self.net(enrollment.unsqueeze(1))
        return hidden.mean(dim=-1)


class FiLMTemporalBlock(nn.Module):
    def __init__(self, channels: int, speaker_dim: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(8, channels)
        self.to_scale_shift = nn.Linear(speaker_dim, channels * 2)

    def forward(self, x: torch.Tensor, speaker_embed: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(speaker_embed).chunk(2, dim=-1)
        h = self.depthwise(x)
        h = self.pointwise(h)
        h = self.norm(h)
        h = h * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = F.gelu(h)
        return x + h


class WaveformTSE(nn.Module):
    def __init__(
        self,
        base_channels: int = 128,
        speaker_dim: int = 128,
        num_blocks: int = 8,
    ) -> None:
        super().__init__()
        self.speaker_encoder = EnrollmentEncoder1D(speaker_dim)
        self.in_proj = nn.Conv1d(1, base_channels, kernel_size=15, padding=7)
        dilations = [2 ** (idx % 4) for idx in range(num_blocks)]
        self.blocks = nn.ModuleList(
            [FiLMTemporalBlock(base_channels, speaker_dim, dilation=dilation) for dilation in dilations]
        )
        self.mid = nn.Conv1d(base_channels, base_channels, kernel_size=7, padding=3)
        self.out_proj = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(base_channels, 1, kernel_size=1),
        )

    def forward(self, mixture: torch.Tensor, enrollment: torch.Tensor) -> torch.Tensor:
        speaker_embed = self.speaker_encoder(enrollment)
        hidden = self.in_proj(mixture.unsqueeze(1))
        for block in self.blocks:
            hidden = block(hidden, speaker_embed)
        hidden = F.gelu(self.mid(hidden))
        estimate = self.out_proj(hidden).squeeze(1)
        return torch.tanh(estimate)
