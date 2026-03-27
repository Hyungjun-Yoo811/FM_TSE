from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try: from pesq import pesq as pesq_fn
except ImportError: pesq_fn = None



def si_sdr(estimation: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    reference_energy = torch.sum(reference * reference, dim=-1, keepdim=True) + eps
    projection = torch.sum(estimation * reference, dim=-1, keepdim=True) * reference / reference_energy
    noise = estimation - projection
    ratio = (torch.sum(projection * projection, dim=-1) + eps) / (torch.sum(noise * noise, dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def si_snr(estimation: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    estimation = estimation - estimation.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    return si_sdr(estimation, reference, eps=eps)


def snr(estimation: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    signal = torch.sum(reference * reference, dim=-1)
    noise = torch.sum((estimation - reference) ** 2, dim=-1)
    return 10.0 * torch.log10((signal + eps) / (noise + eps))


def improvement(
    metric_fn,
    estimation: torch.Tensor,
    reference: torch.Tensor,
    baseline: torch.Tensor,
) -> torch.Tensor:
    return metric_fn(estimation, reference) - metric_fn(baseline, reference)


def mel_l1(estimation: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, estimation.ndim))
    return torch.mean(torch.abs(estimation - reference), dim=dims)


def spectral_convergence(estimation: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    numerator = torch.linalg.vector_norm((reference - estimation).reshape(estimation.shape[0], -1), dim=-1)
    denominator = torch.linalg.vector_norm(reference.reshape(reference.shape[0], -1), dim=-1) + eps
    return numerator / denominator


def mel_frame_cosine_similarity(estimation: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    estimation_flat = estimation.transpose(1, 2)
    reference_flat = reference.transpose(1, 2)
    cosine = F.cosine_similarity(estimation_flat, reference_flat, dim=-1)
    return cosine.mean(dim=-1)


def pesq_score(estimation: torch.Tensor, reference: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if pesq_fn is None:
        return torch.full((estimation.shape[0],), float("nan"), device=estimation.device)

    mode = "wb" if sample_rate == 16000 else "nb"
    scores = []
    estimation_cpu = estimation.detach().cpu().float()
    reference_cpu = reference.detach().cpu().float()

    for est, ref in zip(estimation_cpu, reference_cpu):
        try:
            score = pesq_fn(sample_rate, ref.numpy(), est.numpy(), mode)
        except Exception:
            score = math.nan
        scores.append(score)

    return torch.tensor(scores, dtype=estimation.dtype, device=estimation.device)
