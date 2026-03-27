from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_training_curves(history: dict[str, list[float]], output_path: str | Path) -> None:
    output_path = _ensure_parent(output_path)
    metrics = [(name, values) for name, values in history.items() if values]
    if not metrics:
        return

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3.2 * len(metrics)), squeeze=False)
    for axis, (name, values) in zip(axes[:, 0], metrics):
        axis.plot(np.arange(1, len(values) + 1), values, linewidth=2)
        axis.set_title(name.replace("_", " ").title())
        axis.set_xlabel("Step" if "train" in name else "Epoch")
        axis.set_ylabel(name)
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_mel_panel(
    mel_tensors: dict[str, torch.Tensor],
    output_path: str | Path,
    title: str,
) -> None:
    output_path = _ensure_parent(output_path)
    items = list(mel_tensors.items())
    fig, axes = plt.subplots(1, len(items), figsize=(4.5 * len(items), 4), squeeze=False)

    for axis, (name, tensor) in zip(axes[0], items):
        image = tensor.detach().cpu().float().numpy()
        axis.imshow(image, origin="lower", aspect="auto", interpolation="nearest")
        axis.set_title(name)
        axis.set_xlabel("Frame")
        axis.set_ylabel("Mel Bin")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_vector_field_panel(
    field_tensors: dict[str, torch.Tensor],
    output_path: str | Path,
    title: str,
) -> None:
    output_path = _ensure_parent(output_path)
    items = list(field_tensors.items())
    fig, axes = plt.subplots(1, len(items), figsize=(4.8 * len(items), 4), squeeze=False)

    arrays = [tensor.detach().cpu().float().numpy() for _, tensor in items]
    vmax = max(float(np.max(np.abs(array))) for array in arrays) if arrays else 1.0
    vmax = max(vmax, 1e-6)

    for axis, (name, _), image in zip(axes[0], items, arrays):
        if "error" in name.lower():
            handle = axis.imshow(image, origin="lower", aspect="auto", interpolation="nearest", cmap="magma")
        else:
            handle = axis.imshow(
                image,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
            )
        axis.set_title(name)
        axis.set_xlabel("Frame")
        axis.set_ylabel("Mel Bin")
        fig.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_flow_trajectory_plot(
    flow_history: Iterable[torch.Tensor],
    output_path: str | Path,
    title: str,
) -> None:
    output_path = _ensure_parent(output_path)
    steps = list(flow_history)
    if not steps:
        return

    means = []
    stds = []
    deltas = [0.0]
    prev = None
    for state in steps:
        state_cpu = state.detach().cpu().float()
        means.append(state_cpu.mean().item())
        stds.append(state_cpu.std().item())
        if prev is not None:
            deltas.append(torch.mean(torch.abs(state_cpu - prev)).item())
        prev = state_cpu

    x = np.arange(len(steps))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(x, means, label="mean", linewidth=2)
    axes[0].plot(x, stds, label="std", linewidth=2)
    axes[0].set_ylabel("Activation")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, deltas, color="tab:red", linewidth=2)
    axes[1].set_xlabel("Euler Step")
    axes[1].set_ylabel("Mean |delta|")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
