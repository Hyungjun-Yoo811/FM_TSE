from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def _normalize_gpu_ids(values: Iterable) -> list[int]:
    gpu_ids = []
    for value in values:
        if isinstance(value, str):
            value = value.strip().lower()
            if value.startswith("cuda:"):
                value = value.split(":", 1)[1]
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index not in gpu_ids:
            gpu_ids.append(index)
    return gpu_ids


def resolve_device(device_config) -> tuple[torch.device, list[int]]:
    if isinstance(device_config, int):
        if torch.cuda.is_available() and 0 <= device_config < torch.cuda.device_count():
            return torch.device(f"cuda:{device_config}"), [device_config]
        return torch.device("cpu"), []

    if isinstance(device_config, (list, tuple)):
        gpu_ids = [idx for idx in _normalize_gpu_ids(device_config) if 0 <= idx < torch.cuda.device_count()]
        if torch.cuda.is_available() and gpu_ids:
            return torch.device(f"cuda:{gpu_ids[0]}"), gpu_ids
        return torch.device("cpu"), []

    if isinstance(device_config, str):
        name = device_config.strip().lower()
        if name == "cpu":
            return torch.device("cpu"), []
        if name == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda:0"), [0]
            return torch.device("cpu"), []
        if name.isdigit():
            index = int(name)
            if torch.cuda.is_available() and 0 <= index < torch.cuda.device_count():
                return torch.device(f"cuda:{index}"), [index]
            return torch.device("cpu"), []
        if name.startswith("cuda:"):
            try:
                index = int(name.split(":", 1)[1])
            except ValueError:
                return torch.device("cpu"), []
            if torch.cuda.is_available() and 0 <= index < torch.cuda.device_count():
                return torch.device(f"cuda:{index}"), [index]
            return torch.device("cpu"), []

    return torch.device("cpu"), []


def maybe_wrap_model(model: nn.Module, device: torch.device, gpu_ids: list[int]) -> nn.Module:
    model = model.to(device)
    if device.type == "cuda" and len(gpu_ids) > 1:
        return nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def model_state_dict(model: nn.Module) -> dict:
    return unwrap_model(model).state_dict()


def load_model_state(model: nn.Module, state_dict: dict) -> None:
    unwrap_model(model).load_state_dict(state_dict)
