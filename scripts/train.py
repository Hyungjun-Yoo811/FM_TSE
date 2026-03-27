from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fm_tse.data.features import MelSpectrogramConfig, MelSpectrogramTransform
from src.fm_tse.data.pipeline import build_dataloader
from src.fm_tse.models.flow_matching import euler_sample, sample_flow_matching_batch
from src.fm_tse.models.networks import FlowMatchingTSE
from src.fm_tse.utils.config import load_config
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
from src.fm_tse.utils.visualization import (
    save_flow_trajectory_plot,
    save_mel_panel,
    save_training_curves,
    save_vector_field_panel,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def build_feature_extractor(config: Dict, device: torch.device) -> MelSpectrogramTransform:
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


def run_validation(model, valid_loader, device, steps: int, feature_extractor, output_dir: Path, epoch: int) -> Dict[str, float]:
    model.eval()
    flow_mse_total = 0.0
    sample_mse_total = 0.0
    mel_l1_total = 0.0
    spectral_convergence_total = 0.0
    cosine_total = 0.0
    si_sdr_total = 0.0
    si_sdri_total = 0.0
    si_snr_total = 0.0
    si_snri_total = 0.0
    snr_total = 0.0
    pesq_total = 0.0
    batches = 0
    pesq_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):
            batch = move_batch(batch, device)
            x_t, time, flow_target, mixture, enrollment, target = sample_flow_matching_batch(batch, feature_extractor)
            pred = model(x_t, time, mixture, enrollment)
            flow_mse_total += torch.mean((pred - flow_target) ** 2).item()

            sampled, mixture_mel, enrollment_mel, history = euler_sample(
                model,
                batch["mixture"],
                batch["enrollment"],
                steps=steps,
                feature_extractor=feature_extractor,
                return_history=True,
            )
            sample_mse_total += torch.mean((sampled - target) ** 2).item()
            mel_l1_total += mel_l1(sampled, target).mean().item()
            spectral_convergence_total += spectral_convergence(sampled, target).mean().item()
            cosine_total += mel_frame_cosine_similarity(sampled, target).mean().item()
            estimate_wave = feature_extractor.griffin_lim(sampled, length=batch["target"].shape[-1])
            target_wave = batch["target"]
            mixture_wave = batch["mixture"]
            si_sdr_total += si_sdr(estimate_wave, target_wave).mean().item()
            si_sdri_total += improvement(si_sdr, estimate_wave, target_wave, mixture_wave).mean().item()
            si_snr_total += si_snr(estimate_wave, target_wave).mean().item()
            si_snri_total += improvement(si_snr, estimate_wave, target_wave, mixture_wave).mean().item()
            snr_total += snr(estimate_wave, target_wave).mean().item()
            pesq_value = pesq_score(estimate_wave, target_wave, sample_rate=feature_extractor.config.sample_rate)
            if not torch.isnan(pesq_value).all():
                pesq_total += torch.nanmean(pesq_value).item()
                pesq_batches += 1
            batches += 1

            if batch_idx == 0:
                plots_dir = output_dir / "plots"
                flow_error = torch.abs(pred - flow_target)
                save_mel_panel(
                    {
                        "mixture": mixture_mel[0],
                        "target": target[0],
                        "estimate": sampled[0],
                        "enrollment": enrollment_mel[0],
                    },
                    plots_dir / "validation_mels.png",
                    title=f"Validation Mel Spectrograms (Epoch {epoch})",
                )
                save_mel_panel(
                    {
                        "mixture": mixture_mel[0],
                        "target": target[0],
                        "estimate": sampled[0],
                        "enrollment": enrollment_mel[0],
                    },
                    plots_dir / f"validation_mels_epoch_{epoch:03d}.png",
                    title=f"Validation Mel Spectrograms (Epoch {epoch})",
                )
                save_vector_field_panel(
                    {
                        "state_xt": x_t[0],
                        "flow_target": flow_target[0],
                        "pred_flow": pred[0],
                        "flow_error": flow_error[0],
                    },
                    plots_dir / "validation_vector_field.png",
                    title=f"Validation Vector Field (Epoch {epoch})",
                )
                save_vector_field_panel(
                    {
                        "state_xt": x_t[0],
                        "flow_target": flow_target[0],
                        "pred_flow": pred[0],
                        "flow_error": flow_error[0],
                    },
                    plots_dir / f"validation_vector_field_epoch_{epoch:03d}.png",
                    title=f"Validation Vector Field (Epoch {epoch})",
                )
                save_flow_trajectory_plot(
                    [state[0] for state in history],
                    plots_dir / "validation_flow.png",
                    title=f"Validation Flow Trajectory (Epoch {epoch})",
                )
                save_flow_trajectory_plot(
                    [state[0] for state in history],
                    plots_dir / f"validation_flow_epoch_{epoch:03d}.png",
                    title=f"Validation Flow Trajectory (Epoch {epoch})",
                )

    return {
        "flow_mse": flow_mse_total / max(batches, 1),
        "sample_mse": sample_mse_total / max(batches, 1),
        "mel_l1": mel_l1_total / max(batches, 1),
        "spectral_convergence": spectral_convergence_total / max(batches, 1),
        "frame_cosine_similarity": cosine_total / max(batches, 1),
        "si_sdr": si_sdr_total / max(batches, 1),
        "si_sdri": si_sdri_total / max(batches, 1),
        "si_snr": si_snr_total / max(batches, 1),
        "si_snri": si_snri_total / max(batches, 1),
        "snr": snr_total / max(batches, 1),
        "pesq": pesq_total / max(pesq_batches, 1) if pesq_batches > 0 else float("nan"),
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, config: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/librimix/baseline.json")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    feature_extractor = build_feature_extractor(config, device)

    train_loader = build_dataloader(config, "train")
    valid_loader = build_dataloader(config, "valid")
    model = FlowMatchingTSE(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=1e-4,
    )

    output_dir = Path(config["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_flow_mse": [],
        "valid_flow_mse": [],
        "valid_sample_mse": [],
        "valid_mel_l1": [],
        "valid_spectral_convergence": [],
        "valid_frame_cosine_similarity": [],
        "valid_si_sdr": [],
        "valid_si_sdri": [],
        "valid_si_snr": [],
        "valid_si_snri": [],
        "valid_snr": [],
        "valid_pesq": [],
    }

    global_step = 0
    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            x_t, time, flow_target, mixture, enrollment, target = sample_flow_matching_batch(batch, feature_extractor)

            pred = model(x_t, time, mixture, enrollment)
            loss = torch.mean((pred - flow_target) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])
            optimizer.step()

            running_loss += loss.item()
            history["train_flow_mse"].append(loss.item())
            global_step += 1

            if global_step % config["train"]["log_every"] == 0:
                avg_loss = running_loss / config["train"]["log_every"]
                print(f"[train] epoch={epoch} step={global_step} flow_mse={avg_loss:.6f}")
                running_loss = 0.0

            if epoch == 1 and batch_idx == 0:
                save_mel_panel(
                    {
                        "mixture": mixture[0],
                        "target": target[0],
                        "state_xt": x_t[0],
                        "enrollment": enrollment[0],
                    },
                    plot_dir / "train_batch_mels.png",
                    title="Train Batch Mel Views",
                )
            if batch_idx == 0:
                flow_error = torch.abs(pred - flow_target)
                save_vector_field_panel(
                    {
                        "state_xt": x_t[0],
                        "flow_target": flow_target[0],
                        "pred_flow": pred[0],
                        "flow_error": flow_error[0],
                    },
                    plot_dir / "train_vector_field.png",
                    title=f"Train Vector Field (Epoch {epoch})",
                )
                save_vector_field_panel(
                    {
                        "state_xt": x_t[0],
                        "flow_target": flow_target[0],
                        "pred_flow": pred[0],
                        "flow_error": flow_error[0],
                    },
                    plot_dir / f"train_vector_field_epoch_{epoch:03d}.png",
                    title=f"Train Vector Field (Epoch {epoch})",
                )

        metrics = run_validation(
            model,
            valid_loader,
            device,
            config["sampling"]["steps"],
            feature_extractor,
            output_dir,
            epoch,
        )
        history["valid_flow_mse"].append(metrics["flow_mse"])
        history["valid_sample_mse"].append(metrics["sample_mse"])
        history["valid_mel_l1"].append(metrics["mel_l1"])
        history["valid_spectral_convergence"].append(metrics["spectral_convergence"])
        history["valid_frame_cosine_similarity"].append(metrics["frame_cosine_similarity"])
        history["valid_si_sdr"].append(metrics["si_sdr"])
        history["valid_si_sdri"].append(metrics["si_sdri"])
        history["valid_si_snr"].append(metrics["si_snr"])
        history["valid_si_snri"].append(metrics["si_snri"])
        history["valid_snr"].append(metrics["snr"])
        history["valid_pesq"].append(metrics["pesq"])
        save_training_curves(history, plot_dir / "training_curves.png")

        print(
            f"[valid] epoch={epoch} flow_mse={metrics['flow_mse']:.6f} "
            f"sample_mse={metrics['sample_mse']:.6f} "
            f"mel_l1={metrics['mel_l1']:.6f} "
            f"spectral_convergence={metrics['spectral_convergence']:.6f} "
            f"frame_cosine_similarity={metrics['frame_cosine_similarity']:.6f} "
            f"si_sdr={metrics['si_sdr']:.3f} "
            f"si_sdri={metrics['si_sdri']:.3f} "
            f"si_snr={metrics['si_snr']:.3f} "
            f"si_snri={metrics['si_snri']:.3f} "
            f"snr={metrics['snr']:.3f} "
            f"pesq={metrics['pesq']:.3f}"
        )

        latest_path = checkpoint_dir / "latest.pt"
        save_checkpoint(latest_path, model, optimizer, epoch, config)
        if epoch % config["train"]["save_every"] == 0:
            save_checkpoint(checkpoint_dir / f"epoch_{epoch}.pt", model, optimizer, epoch, config)


if __name__ == "__main__":
    main()
