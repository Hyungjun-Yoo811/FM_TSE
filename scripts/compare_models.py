from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fm_tse.data.features import MelSpectrogramConfig, MelSpectrogramTransform
from src.fm_tse.data.pipeline import build_dataloader
from src.fm_tse.models.flow_matching import euler_sample
from src.fm_tse.models.networks import FlowMatchingTSE
from src.fm_tse.models.stft_tse import STFTMaskTSE
from src.fm_tse.models.waveform_tse import WaveformTSE
from src.fm_tse.utils.config import load_config
from src.fm_tse.utils.device import load_model_state, maybe_wrap_model, resolve_device
from src.fm_tse.utils.metrics import pesq_score, si_sdr


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


def move_batch(batch, device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


class STFTCodec:
    def __init__(self, config, device: torch.device) -> None:
        stft_cfg = config["stft"]
        self.n_fft = stft_cfg["n_fft"]
        self.hop_length = stft_cfg["hop_length"]
        self.win_length = stft_cfg["win_length"]
        self.window = torch.hann_window(self.win_length, device=device)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, return_complex=True)
        return torch.stack([spec.real, spec.imag], dim=1)

    def decode(self, spec_2ch: torch.Tensor, length: int) -> torch.Tensor:
        spec = torch.complex(spec_2ch[:, 0], spec_2ch[:, 1])
        return torch.istft(spec, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, length=length)


def evaluate_mel(mel_config: dict, checkpoint_path: str, split: str, device: torch.device) -> dict[str, float]:
    config = load_config(mel_config)
    loader = build_dataloader(config, split)
    feature_extractor = build_feature_extractor(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = maybe_wrap_model(FlowMatchingTSE(**checkpoint["config"]["model"]), device, [device.index] if device.type == "cuda" else [])
    load_model_state(model, checkpoint["model"])
    model.eval()

    totals = {"si_sdr": 0.0, "pesq": 0.0}
    batches = 0
    pesq_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            estimate_mel, _, _, _ = euler_sample(
                model,
                batch["mixture"],
                batch["enrollment"],
                steps=config["sampling"]["steps"],
                feature_extractor=feature_extractor,
                return_history=True,
            )
            estimate = feature_extractor.griffin_lim(estimate_mel, length=batch["target"].shape[-1])
            totals["si_sdr"] += si_sdr(estimate, batch["target"]).mean().item()
            pesq_value = pesq_score(estimate, batch["target"], config["sample_rate"])
            if not torch.isnan(pesq_value).all():
                totals["pesq"] += torch.nanmean(pesq_value).item()
                pesq_batches += 1
            batches += 1

    return {
        "si_sdr": totals["si_sdr"] / max(batches, 1),
        "pesq": totals["pesq"] / max(pesq_batches, 1) if pesq_batches > 0 else float("nan"),
    }


def evaluate_waveform(config_path: str, checkpoint_path: str, split: str, device: torch.device) -> dict[str, float]:
    config = load_config(config_path)
    loader = build_dataloader(config, split)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = maybe_wrap_model(WaveformTSE(**checkpoint["config"]["model"]), device, [device.index] if device.type == "cuda" else [])
    load_model_state(model, checkpoint["model"])
    model.eval()

    totals = {"si_sdr": 0.0, "pesq": 0.0}
    batches = 0
    pesq_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            estimate = model(batch["mixture"], batch["enrollment"])
            totals["si_sdr"] += si_sdr(estimate, batch["target"]).mean().item()
            pesq_value = pesq_score(estimate, batch["target"], config["sample_rate"])
            if not torch.isnan(pesq_value).all():
                totals["pesq"] += torch.nanmean(pesq_value).item()
                pesq_batches += 1
            batches += 1

    return {
        "si_sdr": totals["si_sdr"] / max(batches, 1),
        "pesq": totals["pesq"] / max(pesq_batches, 1) if pesq_batches > 0 else float("nan"),
    }


def evaluate_stft(config_path: str, checkpoint_path: str, split: str, device: torch.device) -> dict[str, float]:
    config = load_config(config_path)
    loader = build_dataloader(config, split)
    codec = STFTCodec(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = maybe_wrap_model(STFTMaskTSE(**checkpoint["config"]["model"]), device, [device.index] if device.type == "cuda" else [])
    load_model_state(model, checkpoint["model"])
    model.eval()

    totals = {"si_sdr": 0.0, "pesq": 0.0}
    batches = 0
    pesq_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            estimate_spec = model(codec.encode(batch["mixture"]), codec.encode(batch["enrollment"]))
            estimate = codec.decode(estimate_spec, length=batch["target"].shape[-1])
            totals["si_sdr"] += si_sdr(estimate, batch["target"]).mean().item()
            pesq_value = pesq_score(estimate, batch["target"], config["sample_rate"])
            if not torch.isnan(pesq_value).all():
                totals["pesq"] += torch.nanmean(pesq_value).item()
                pesq_batches += 1
            batches += 1

    return {
        "si_sdr": totals["si_sdr"] / max(batches, 1),
        "pesq": totals["pesq"] / max(pesq_batches, 1) if pesq_batches > 0 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mel-config", type=str, default="configs/librimix/baseline.json")
    parser.add_argument("--mel-checkpoint", type=str, required=True)
    parser.add_argument("--waveform-config", type=str, default="configs/librimix/waveform.json")
    parser.add_argument("--waveform-checkpoint", type=str, required=True)
    parser.add_argument("--stft-config", type=str, default="configs/librimix/stft.json")
    parser.add_argument("--stft-checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--output", type=str, default="outputs/librimix/comparison_three_way.json")
    args = parser.parse_args()

    waveform_config = load_config(args.waveform_config)
    device, _ = resolve_device(waveform_config["device"])

    mel_metrics = evaluate_mel(args.mel_config, args.mel_checkpoint, args.split, device)
    waveform_metrics = evaluate_waveform(args.waveform_config, args.waveform_checkpoint, args.split, device)
    stft_metrics = evaluate_stft(args.stft_config, args.stft_checkpoint, args.split, device)
    comparison = {
        "split": args.split,
        "mel": mel_metrics,
        "waveform": waveform_metrics,
        "stft": stft_metrics,
        "delta_waveform_minus_mel": {
            "si_sdr": waveform_metrics["si_sdr"] - mel_metrics["si_sdr"],
            "pesq": waveform_metrics["pesq"] - mel_metrics["pesq"],
        },
        "delta_stft_minus_mel": {
            "si_sdr": stft_metrics["si_sdr"] - mel_metrics["si_sdr"],
            "pesq": stft_metrics["pesq"] - mel_metrics["pesq"],
        },
        "delta_stft_minus_waveform": {
            "si_sdr": stft_metrics["si_sdr"] - waveform_metrics["si_sdr"],
            "pesq": stft_metrics["pesq"] - waveform_metrics["pesq"],
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    print(f"Saved comparison to: {output_path}")


if __name__ == "__main__":
    main()
