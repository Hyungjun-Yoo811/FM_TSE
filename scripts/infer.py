from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fm_tse.data.features import MelSpectrogramConfig, MelSpectrogramTransform
from src.fm_tse.data.audio import save_wav
from src.fm_tse.data.pipeline import build_dataset
from src.fm_tse.models.flow_matching import euler_sample
from src.fm_tse.models.networks import FlowMatchingTSE
from src.fm_tse.utils.config import load_config
from src.fm_tse.utils.device import resolve_device
from src.fm_tse.utils.metrics import (
    improvement,
    mel_frame_cosine_similarity,
    mel_l1,
    pesq_score,
    si_sdr,
    si_snr,
    snr,
    spectral_convergence,
)
from src.fm_tse.utils.visualization import save_flow_trajectory_plot, save_mel_panel
def build_feature_extractor(config, device: torch.device) -> MelSpectrogramTransform:
    feature_cfg = config["features"]
    mel_config = MelSpectrogramConfig(
        sample_rate=config["sample_rate"],
        n_fft=feature_cfg["n_fft"],
        hop_length=feature_cfg["hop_length"],
        win_length=feature_cfg["win_length"],
        n_mels=feature_cfg["n_mels"],
        f_min=feature_cfg.get("f_min", 0.0),
        f_max=feature_cfg.get("f_max"),
        log_epsilon=feature_cfg.get("log_epsilon", 1e-5),
    )
    return MelSpectrogramTransform(mel_config).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/librimix/baseline.json")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config["device"])
    feature_extractor = build_feature_extractor(config, device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = FlowMatchingTSE(**checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = build_dataset(config, args.split, inference=True)
    batch = dataset[args.index]
    mixture = batch["mixture"].unsqueeze(0).to(device)
    enrollment = batch["enrollment"].unsqueeze(0).to(device)
    target = batch["target"].unsqueeze(0).to(device)

    with torch.no_grad():
        estimate, mixture_mel, enrollment_mel, history = euler_sample(
            model,
            mixture,
            enrollment,
            steps=config["sampling"]["steps"],
            feature_extractor=feature_extractor,
            return_history=True,
        )
        target_mel = feature_extractor(target)
        estimate_wave = feature_extractor.griffin_lim(estimate, length=target.shape[-1])

    output_dir = Path(config["output_dir"]) / "samples"
    save_mel_panel(
        {
            "mixture": mixture_mel[0],
            "target": target_mel[0],
            "estimate": estimate[0],
            "enrollment": enrollment_mel[0],
        },
        output_dir / "inference_mels.png",
        title=f"Inference Mel Spectrograms ({args.split}:{args.index})",
    )
    save_flow_trajectory_plot(
        [state[0] for state in history],
        output_dir / "inference_flow.png",
        title="Inference Flow Trajectory",
    )

    torch.save(
        {
            "mixture_mel": mixture_mel.squeeze(0).cpu(),
            "target_mel": target_mel.squeeze(0).cpu(),
            "estimate_mel": estimate.squeeze(0).cpu(),
            "enrollment_mel": enrollment_mel.squeeze(0).cpu(),
            "estimate_waveform": estimate_wave.squeeze(0).cpu(),
        },
        output_dir / "inference_tensors.pt",
    )
    save_wav(
        output_dir / f"inference_{args.split}_{args.index}_estimate.wav",
        estimate_wave.squeeze(0).cpu().numpy(),
        config["sample_rate"],
    )
    save_wav(
        output_dir / f"inference_{args.split}_{args.index}_mixture.wav",
        mixture.squeeze(0).cpu().numpy(),
        config["sample_rate"],
    )
    save_wav(
        output_dir / f"inference_{args.split}_{args.index}_target.wav",
        target.squeeze(0).cpu().numpy(),
        config["sample_rate"],
    )
    save_wav(
        output_dir / f"inference_{args.split}_{args.index}_enrollment.wav",
        enrollment.squeeze(0).cpu().numpy(),
        config["sample_rate"],
    )

    print(
        "[infer] "
        f"mel_l1={mel_l1(estimate, target_mel).mean().item():.6f} "
        f"spectral_convergence={spectral_convergence(estimate, target_mel).mean().item():.6f} "
        f"frame_cosine_similarity={mel_frame_cosine_similarity(estimate, target_mel).mean().item():.6f} "
        f"si_sdr={si_sdr(estimate_wave, target).mean().item():.3f} "
        f"si_sdri={improvement(si_sdr, estimate_wave, target, mixture).mean().item():.3f} "
        f"si_snr={si_snr(estimate_wave, target).mean().item():.3f} "
        f"si_snri={improvement(si_snr, estimate_wave, target, mixture).mean().item():.3f} "
        f"snr={snr(estimate_wave, target).mean().item():.3f} "
        f"pesq={torch.nanmean(pesq_score(estimate_wave, target, config['sample_rate'])).item():.3f}"
    )
    print(f"Saved mel plots, tensors, and wav files to: {output_dir}")


if __name__ == "__main__":
    main()
