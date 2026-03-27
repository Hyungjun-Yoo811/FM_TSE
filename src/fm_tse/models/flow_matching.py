from __future__ import annotations

from typing import Dict, Tuple

import torch


def _to_mel_batch(
    batch: Dict[str, torch.Tensor],
    feature_extractor,
) -> Dict[str, torch.Tensor]:
    return {
        "mixture_mel": feature_extractor(batch["mixture"]),
        "target_mel": feature_extractor(batch["target"]),
        "enrollment_mel": feature_extractor(batch["enrollment"]),
    }


def sample_flow_matching_batch(
    batch: Dict[str, torch.Tensor],
    feature_extractor,
) -> Tuple[torch.Tensor, ...]:
    mel_batch = _to_mel_batch(batch, feature_extractor)
    target = mel_batch["target_mel"]
    mixture = mel_batch["mixture_mel"]
    enrollment = mel_batch["enrollment_mel"]

    noise = torch.randn_like(target)
    time = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
    time_view = time[:, None, None]

    x_t = (1.0 - time_view) * noise + time_view * target
    flow_target = target - noise
    return x_t, time, flow_target, mixture, enrollment, target


@torch.no_grad()
def euler_sample(
    model,
    mixture: torch.Tensor,
    enrollment: torch.Tensor,
    steps: int,
    feature_extractor,
    return_history: bool = False,
):
    mixture_mel = feature_extractor(mixture)
    enrollment_mel = feature_extractor(enrollment)
    state = torch.randn_like(mixture_mel)
    step_size = 1.0 / steps
    batch_size = mixture_mel.shape[0]
    history = [state.detach().cpu()] if return_history else None

    for step in range(steps):
        time = torch.full(
            (batch_size,),
            fill_value=step / steps,
            device=mixture.device,
            dtype=mixture_mel.dtype,
        )
        velocity = model(state, time, mixture_mel, enrollment_mel)
        state = state + step_size * velocity
        if return_history:
            history.append(state.detach().cpu())

    if return_history:
        return state, mixture_mel, enrollment_mel, history
    return state
