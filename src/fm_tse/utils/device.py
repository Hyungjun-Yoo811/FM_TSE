from __future__ import annotations

import torch


def resolve_device(device_config) -> torch.device:
    if isinstance(device_config, int):
        if torch.cuda.is_available() and 0 <= device_config < torch.cuda.device_count():
            return torch.device(f"cuda:{device_config}")
        return torch.device("cpu")

    if isinstance(device_config, str):
        name = device_config.strip().lower()
        if name == "cpu":
            return torch.device("cpu")
        if name == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name.isdigit():
            index = int(name)
            if torch.cuda.is_available() and 0 <= index < torch.cuda.device_count():
                return torch.device(f"cuda:{index}")
            return torch.device("cpu")
        if name.startswith("cuda:"):
            try:
                index = int(name.split(":", 1)[1])
            except ValueError:
                return torch.device("cpu")
            if torch.cuda.is_available() and 0 <= index < torch.cuda.device_count():
                return torch.device(f"cuda:{index}")
            return torch.device("cpu")

    return torch.device("cpu")
