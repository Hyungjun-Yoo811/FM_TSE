from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnrollmentEncoder2D(nn.Module):
    def __init__(self, speaker_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=(1, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(1, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(128, speaker_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, enrollment_spec: torch.Tensor) -> torch.Tensor:
        hidden = self.net(enrollment_spec)
        return hidden.mean(dim=(-1, -2))


class FiLMResidual2D(nn.Module):
    def __init__(self, channels: int, speaker_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.to_scale_shift = nn.Linear(speaker_dim, channels * 2)

    def forward(self, x: torch.Tensor, speaker_embed: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(speaker_embed).chunk(2, dim=-1)
        h = self.norm1(self.conv1(x))
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.gelu(h)
        h = self.norm2(self.conv2(h))
        h = F.gelu(h)
        return x + h


class STFTMaskTSE(nn.Module):
    def __init__(
        self,
        base_channels: int = 48,
        speaker_dim: int = 128,
        num_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.speaker_encoder = EnrollmentEncoder2D(speaker_dim)
        self.in_proj = nn.Conv2d(2, base_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([FiLMResidual2D(base_channels, speaker_dim) for _ in range(num_blocks)])
        self.mid = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.mask_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, 2, kernel_size=1),
        )

    def forward(self, mixture_spec: torch.Tensor, enrollment_spec: torch.Tensor) -> torch.Tensor:
        speaker_embed = self.speaker_encoder(enrollment_spec)
        hidden = self.in_proj(mixture_spec)
        for block in self.blocks:
            hidden = block(hidden, speaker_embed)
        hidden = F.gelu(self.mid(hidden))
        mask = torch.tanh(self.mask_head(hidden))
        return mask * mixture_spec
