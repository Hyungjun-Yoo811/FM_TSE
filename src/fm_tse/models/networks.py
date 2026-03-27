from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(time: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    freq = torch.exp(
        torch.arange(half_dim, device=time.device, dtype=time.dtype)
        * (-math.log(10000.0) / max(half_dim - 1, 1))
    )
    angles = time[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class SpeakerEncoder(nn.Module):
    def __init__(self, speaker_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(1, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(128, speaker_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, enrollment: torch.Tensor) -> torch.Tensor:
        hidden = self.net(enrollment.unsqueeze(1))
        return hidden.mean(dim=(-1, -2))


class FiLMResidualBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.to_scale_shift = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm1(self.conv1(x))
        h = h * (1.0 + scale) + shift
        h = F.gelu(h)
        h = self.norm2(self.conv2(h))
        h = F.gelu(h)
        return x + h


class FlowMatchingTSE(nn.Module):
    def __init__(
        self,
        base_channels: int = 48,
        speaker_dim: int = 128,
        time_dim: int = 128,
        num_blocks: int = 8,
    ) -> None:
        super().__init__()
        cond_dim = speaker_dim + time_dim

        self.speaker_encoder = SpeakerEncoder(speaker_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_proj = nn.Conv2d(2, base_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(base_channels, cond_dim) for _ in range(num_blocks)]
        )
        self.mid = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.out_proj = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, 1, kernel_size=1),
        )
        self.time_dim = time_dim

    def _condition(self, time: torch.Tensor, enrollment: torch.Tensor) -> torch.Tensor:
        time_embed = sinusoidal_embedding(time, self.time_dim)
        time_embed = self.time_mlp(time_embed)
        speaker_embed = self.speaker_encoder(enrollment)
        return torch.cat([time_embed, speaker_embed], dim=-1)

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        mixture: torch.Tensor,
        enrollment: torch.Tensor,
    ) -> torch.Tensor:
        cond = self._condition(time, enrollment)
        x = torch.stack([state, mixture], dim=1)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, cond)
        h = F.gelu(self.mid(h))
        out = self.out_proj(h)
        return out.squeeze(1)
