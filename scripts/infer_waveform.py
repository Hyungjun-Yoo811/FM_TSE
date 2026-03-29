from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fm_tse.data.audio import save_wav
from src.fm_tse.data.pipeline import build_dataset
from src.fm_tse.models.waveform_tse import WaveformTSE
from src.fm_tse.utils.config import load_config
from src.fm_tse.utils.device import load_model_state, maybe_wrap_model, resolve_device
from src.fm_tse.utils.metrics import improvement, pesq_score, si_sdr, si_snr, snr
from src.fm_tse.utils.visualization import save_waveform_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/librimix/waveform.json")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    device, gpu_ids = resolve_device(config["device"])

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = maybe_wrap_model(WaveformTSE(**checkpoint["config"]["model"]), device, gpu_ids)
    load_model_state(model, checkpoint["model"])
    model.eval()
    print(f"[device] using {device} gpu_ids={gpu_ids if gpu_ids else 'cpu'}")

    dataset = build_dataset(config, args.split, inference=True)
    batch = dataset[args.index]
    mixture = batch["mixture"].unsqueeze(0).to(device)
    enrollment = batch["enrollment"].unsqueeze(0).to(device)
    target = batch["target"].unsqueeze(0).to(device)

    with torch.no_grad():
        estimate = model(mixture, enrollment)

    output_dir = Path(config["output_dir"]) / "samples"
    save_waveform_panel(
        {
            "mixture": mixture[0],
            "target": target[0],
            "estimate": estimate[0],
            "residual": estimate[0] - target[0],
        },
        output_dir / "inference_waveforms.png",
        title=f"Waveform Inference ({args.split}:{args.index})",
    )
    save_wav(output_dir / f"inference_{args.split}_{args.index}_estimate.wav", estimate[0].cpu().numpy(), config["sample_rate"])
    save_wav(output_dir / f"inference_{args.split}_{args.index}_mixture.wav", mixture[0].cpu().numpy(), config["sample_rate"])
    save_wav(output_dir / f"inference_{args.split}_{args.index}_target.wav", target[0].cpu().numpy(), config["sample_rate"])
    save_wav(output_dir / f"inference_{args.split}_{args.index}_enrollment.wav", enrollment[0].cpu().numpy(), config["sample_rate"])

    print(
        "[infer-waveform] "
        f"si_sdr={si_sdr(estimate, target).mean().item():.3f} "
        f"si_sdri={improvement(si_sdr, estimate, target, mixture).mean().item():.3f} "
        f"si_snr={si_snr(estimate, target).mean().item():.3f} "
        f"si_snri={improvement(si_snr, estimate, target, mixture).mean().item():.3f} "
        f"snr={snr(estimate, target).mean().item():.3f} "
        f"pesq={torch.nanmean(pesq_score(estimate, target, config['sample_rate'])).item():.3f}"
    )
    print(f"Saved waveform plots and wav files to: {output_dir}")


if __name__ == "__main__":
    main()
