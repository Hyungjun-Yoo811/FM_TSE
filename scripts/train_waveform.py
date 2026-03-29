from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fm_tse.data.pipeline import build_dataloader
from src.fm_tse.models.waveform_tse import WaveformTSE
from src.fm_tse.utils.config import load_config
from src.fm_tse.utils.device import resolve_device
from src.fm_tse.utils.metrics import improvement, pesq_score, si_sdr, si_snr, snr
from src.fm_tse.utils.visualization import save_training_curves, save_waveform_panel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def waveform_loss(estimate: torch.Tensor, target: torch.Tensor, config: Dict) -> torch.Tensor:
    loss_cfg = config.get("loss", {})
    l1_term = torch.mean(torch.abs(estimate - target))
    sisdr_term = -si_sdr(estimate, target).mean()
    return loss_cfg.get("l1_weight", 1.0) * l1_term + loss_cfg.get("si_sdr_weight", 0.1) * sisdr_term


def save_training_state(
    path: Path,
    model,
    optimizer,
    epoch: int,
    config: Dict,
    history: Dict[str, list[float]],
    global_step: int,
    best_metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "history": history,
            "global_step": global_step,
            "best_metrics": best_metrics,
        },
        path,
    )


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


def load_training_state(path: Path, model, optimizer):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return {
        "epoch": checkpoint.get("epoch", 0),
        "history": checkpoint.get("history"),
        "global_step": checkpoint.get("global_step"),
        "best_metrics": checkpoint.get("best_metrics"),
    }


def run_validation(model, valid_loader, device: torch.device, output_dir: Path, epoch: int, config: Dict) -> Dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "si_sdr": 0.0,
        "si_sdri": 0.0,
        "si_snr": 0.0,
        "si_snri": 0.0,
        "snr": 0.0,
        "pesq": 0.0,
    }
    batches = 0
    pesq_batches = 0

    with torch.no_grad():
        valid_bar = tqdm(
            enumerate(valid_loader),
            total=len(valid_loader),
            desc=f"valid-waveform epoch {epoch}",
            leave=False,
        )
        for batch_idx, batch in valid_bar:
            batch = move_batch(batch, device)
            estimate = model(batch["mixture"], batch["enrollment"])
            loss = waveform_loss(estimate, batch["target"], config)
            totals["loss"] += loss.item()
            totals["si_sdr"] += si_sdr(estimate, batch["target"]).mean().item()
            totals["si_sdri"] += improvement(si_sdr, estimate, batch["target"], batch["mixture"]).mean().item()
            totals["si_snr"] += si_snr(estimate, batch["target"]).mean().item()
            totals["si_snri"] += improvement(si_snr, estimate, batch["target"], batch["mixture"]).mean().item()
            totals["snr"] += snr(estimate, batch["target"]).mean().item()
            pesq_value = pesq_score(estimate, batch["target"], config["sample_rate"])
            if not torch.isnan(pesq_value).all():
                totals["pesq"] += torch.nanmean(pesq_value).item()
                pesq_batches += 1
            batches += 1
            valid_bar.set_postfix(
                step=f"{batch_idx + 1}/{len(valid_loader)}",
                loss=f"{loss.item():.4f}",
            )

            if batch_idx == 0:
                plots_dir = output_dir / "plots"
                residual = estimate[0] - batch["target"][0]
                save_waveform_panel(
                    {
                        "mixture": batch["mixture"][0],
                        "target": batch["target"][0],
                        "estimate": estimate[0],
                        "residual": residual,
                    },
                    plots_dir / "validation_waveforms.png",
                    title=f"Validation Waveforms (Epoch {epoch})",
                )
                save_waveform_panel(
                    {
                        "mixture": batch["mixture"][0],
                        "target": batch["target"][0],
                        "estimate": estimate[0],
                        "residual": residual,
                    },
                    plots_dir / f"validation_waveforms_epoch_{epoch:03d}.png",
                    title=f"Validation Waveforms (Epoch {epoch})",
                )

    return {
        "loss": totals["loss"] / max(batches, 1),
        "si_sdr": totals["si_sdr"] / max(batches, 1),
        "si_sdri": totals["si_sdri"] / max(batches, 1),
        "si_snr": totals["si_snr"] / max(batches, 1),
        "si_snri": totals["si_snri"] / max(batches, 1),
        "snr": totals["snr"] / max(batches, 1),
        "pesq": totals["pesq"] / max(pesq_batches, 1) if pesq_batches > 0 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/librimix/waveform.json")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(config["device"])

    train_loader = build_dataloader(config, "train")
    valid_loader = build_dataloader(config, "valid")
    model = WaveformTSE(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"], weight_decay=1e-4)

    output_dir = Path(config["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "valid_loss": [],
        "valid_si_sdr": [],
        "valid_si_sdri": [],
        "valid_si_snr": [],
        "valid_si_snri": [],
        "valid_snr": [],
        "valid_pesq": [],
    }

    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best_si_sdr.pt"
    start_epoch = 1
    global_step = 0
    best_metrics = {"si_sdr": float("-inf"), "epoch": 0}

    if latest_path.exists():
        state = load_training_state(latest_path, model, optimizer)
        start_epoch = state["epoch"] + 1
        if state["history"] is not None:
            history = state["history"]
        if state["global_step"] is not None:
            global_step = state["global_step"]
        if state["best_metrics"] is not None:
            best_metrics = state["best_metrics"]
        print(
            f"[resume] loaded {latest_path} from epoch={state['epoch']} "
            f"best_si_sdr={best_metrics.get('si_sdr', float('-inf')):.3f}"
        )

    for epoch in range(start_epoch, config["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"train-waveform epoch {epoch}/{config['train']['epochs']}",
            leave=True,
        )
        for batch_idx, batch in train_bar:
            batch = move_batch(batch, device)
            estimate = model(batch["mixture"], batch["enrollment"])
            loss = waveform_loss(estimate, batch["target"], config)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])
            optimizer.step()

            running_loss += loss.item()
            history["train_loss"].append(loss.item())
            global_step += 1
            train_bar.set_postfix(
                step=f"{batch_idx + 1}/{len(train_loader)}",
                loss=f"{loss.item():.4f}",
            )

            if global_step % config["train"]["log_every"] == 0:
                avg_loss = running_loss / config["train"]["log_every"]
                print(f"[train-waveform] epoch={epoch} step={global_step} loss={avg_loss:.6f}")
                running_loss = 0.0

            if batch_idx == 0:
                residual = estimate[0] - batch["target"][0]
                save_waveform_panel(
                    {
                        "mixture": batch["mixture"][0],
                        "target": batch["target"][0],
                        "estimate": estimate[0],
                        "residual": residual,
                    },
                    plot_dir / "train_waveforms.png",
                    title=f"Train Waveforms (Epoch {epoch})",
                )
                save_waveform_panel(
                    {
                        "mixture": batch["mixture"][0],
                        "target": batch["target"][0],
                        "estimate": estimate[0],
                        "residual": residual,
                    },
                    plot_dir / f"train_waveforms_epoch_{epoch:03d}.png",
                    title=f"Train Waveforms (Epoch {epoch})",
                )

        metrics = run_validation(model, valid_loader, device, output_dir, epoch, config)
        history["valid_loss"].append(metrics["loss"])
        history["valid_si_sdr"].append(metrics["si_sdr"])
        history["valid_si_sdri"].append(metrics["si_sdri"])
        history["valid_si_snr"].append(metrics["si_snr"])
        history["valid_si_snri"].append(metrics["si_snri"])
        history["valid_snr"].append(metrics["snr"])
        history["valid_pesq"].append(metrics["pesq"])
        save_training_curves(history, plot_dir / "training_curves.png")

        print(
            f"[valid-waveform] epoch={epoch} "
            f"loss={metrics['loss']:.6f} "
            f"si_sdr={metrics['si_sdr']:.3f} "
            f"si_sdri={metrics['si_sdri']:.3f} "
            f"si_snr={metrics['si_snr']:.3f} "
            f"si_snri={metrics['si_snri']:.3f} "
            f"snr={metrics['snr']:.3f} "
            f"pesq={metrics['pesq']:.3f}"
        )

        if metrics["si_sdr"] > best_metrics["si_sdr"]:
            best_metrics = {"si_sdr": metrics["si_sdr"], "epoch": epoch}
            save_training_state(best_path, model, optimizer, epoch, config, history, global_step, best_metrics)
            print(f"[best] epoch={epoch} si_sdr={metrics['si_sdr']:.3f} saved={best_path}")

        save_training_state(latest_path, model, optimizer, epoch, config, history, global_step, best_metrics)
        if epoch % config["train"]["save_every"] == 0:
            save_checkpoint(checkpoint_dir / f"epoch_{epoch}.pt", model, optimizer, epoch, config)


if __name__ == "__main__":
    main()
