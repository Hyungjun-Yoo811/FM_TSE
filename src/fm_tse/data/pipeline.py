from __future__ import annotations

from typing import Dict

from torch.utils.data import DataLoader

from src.fm_tse.data.datasets import Libri2MixTargetSpeechDataset, SyntheticTargetSpeechDataset


def build_dataset(config: Dict, split: str, inference: bool = False):
    data_cfg = config["data"]
    dataset_type = data_cfg.get("dataset_type", "synthetic")

    if dataset_type == "synthetic":
        size_key = {
            "train": "train_size",
            "valid": "valid_size",
            "test": "valid_size",
        }[split]
        seed_offset = {
            "train": 0,
            "valid": 10_000,
            "test": 20_000,
        }[split]
        signal_length = data_cfg["signal_length"]
        enrollment_length = data_cfg["enrollment_length"]
        if inference and split != "train":
            signal_length = data_cfg.get("inference_signal_length", signal_length)
            enrollment_length = data_cfg.get("inference_enrollment_length", enrollment_length)
        return SyntheticTargetSpeechDataset(
            size=data_cfg[size_key],
            signal_length=signal_length,
            enrollment_length=enrollment_length,
            sample_rate=config["sample_rate"],
            seed=config["seed"] + seed_offset,
        )

    if dataset_type == "librimix":
        csv_key = {
            "train": "train_csv",
            "valid": "valid_csv",
            "test": "test_csv",
        }[split]
        if split == "train":
            strategy = data_cfg.get("train_target_source_strategy", data_cfg.get("target_source_strategy", "random"))
        else:
            strategy = data_cfg.get(f"{split}_target_source_strategy", "source_1")
        max_items_key = f"{split}_max_items"
        segment_length = data_cfg["segment_length"]
        enrollment_length = data_cfg["enrollment_length"]
        if inference and split != "train":
            segment_length = data_cfg.get("inference_segment_length", segment_length)
            enrollment_length = data_cfg.get("inference_enrollment_length", enrollment_length)
        return Libri2MixTargetSpeechDataset(
            metadata_csv=data_cfg[csv_key],
            librispeech_root=data_cfg["librispeech_root"],
            wham_root=data_cfg.get("wham_root"),
            sample_rate=config["sample_rate"],
            mode=data_cfg.get("mode", "min"),
            mixture_type=data_cfg.get("mixture_type", "mix_both"),
            segment_length=segment_length,
            enrollment_length=enrollment_length,
            target_source_strategy=strategy,
            training=(split == "train"),
            seed=config["seed"] + {"train": 0, "valid": 10_000, "test": 20_000}[split],
            max_items=data_cfg.get(max_items_key),
        )

    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def build_dataloader(config: Dict, split: str) -> DataLoader:
    dataset = build_dataset(config, split)
    train_cfg = config["train"]
    return DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=(split == "train"),
        num_workers=train_cfg["num_workers"],
        drop_last=(split == "train"),
    )
