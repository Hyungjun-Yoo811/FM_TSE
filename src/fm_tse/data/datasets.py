from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from src.fm_tse.data.audio import crop_or_pad, fit_audio_lengths, load_audio, to_tensor


@dataclass
class SpeakerProfile:
    base_f0: float
    formant_scale: float
    vibrato_rate: float
    vibrato_depth: float


class SyntheticTargetSpeechDataset(Dataset):
    def __init__(
        self,
        size: int,
        signal_length: int,
        enrollment_length: int,
        sample_rate: int,
        seed: int = 42,
    ) -> None:
        self.size = size
        self.signal_length = signal_length
        self.enrollment_length = enrollment_length
        self.sample_rate = sample_rate
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def _rng(self, index: int) -> random.Random:
        return random.Random(self.seed + index)

    def _speaker_profile(self, rng: random.Random) -> SpeakerProfile:
        return SpeakerProfile(
            base_f0=rng.uniform(90.0, 260.0),
            formant_scale=rng.uniform(0.8, 1.25),
            vibrato_rate=rng.uniform(3.0, 7.5),
            vibrato_depth=rng.uniform(0.005, 0.03),
        )

    def _synthesize_voice(
        self,
        profile: SpeakerProfile,
        length: int,
        rng: random.Random,
    ) -> torch.Tensor:
        t = torch.linspace(0.0, length / self.sample_rate, steps=length, dtype=torch.float32)
        vibrato = 1.0 + profile.vibrato_depth * torch.sin(2.0 * math.pi * profile.vibrato_rate * t)
        f0 = profile.base_f0 * vibrato
        torch_seed = rng.randint(0, 2**31 - 1)
        generator = torch.Generator().manual_seed(torch_seed)

        phase = 2.0 * math.pi * torch.cumsum(f0 / self.sample_rate, dim=0)
        harmonics = torch.zeros_like(t)
        num_harmonics = rng.randint(4, 8)
        for harmonic_idx in range(1, num_harmonics + 1):
            weight = rng.uniform(0.2, 1.0) / harmonic_idx
            harmonic_scale = 1.0 + 0.08 * profile.formant_scale * harmonic_idx
            harmonics = harmonics + weight * torch.sin(harmonic_scale * harmonic_idx * phase)

        envelope_rate = rng.uniform(1.5, 4.0)
        envelope = 0.55 + 0.45 * torch.sin(2.0 * math.pi * envelope_rate * t + rng.uniform(0, math.pi))
        envelope = envelope.clamp(min=0.05)

        noise = 0.02 * torch.randn(length, generator=generator)
        voiced = envelope * harmonics + noise

        cutoff = int(length * rng.uniform(0.08, 0.18))
        if cutoff > 0:
            fade = torch.linspace(0.0, 1.0, steps=cutoff)
            voiced[:cutoff] *= fade
            voiced[-cutoff:] *= torch.flip(fade, dims=[0])

        voiced = voiced / (voiced.abs().max() + 1e-6)
        gain = rng.uniform(0.5, 0.95)
        return gain * voiced

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        rng = self._rng(index)

        target_profile = self._speaker_profile(rng)
        interferer_profile = self._speaker_profile(rng)

        target = self._synthesize_voice(target_profile, self.signal_length, rng)
        interferer = self._synthesize_voice(interferer_profile, self.signal_length, rng)
        enrollment = self._synthesize_voice(target_profile, self.enrollment_length, rng)

        mixture_snr = rng.uniform(-2.0, 4.0)
        target_energy = target.pow(2).mean().sqrt()
        interferer_energy = interferer.pow(2).mean().sqrt() + 1e-6
        interferer_scale = (target_energy / interferer_energy) * (10.0 ** (-mixture_snr / 20.0))

        mixture = target + interferer_scale * interferer
        mixture = mixture / (mixture.abs().max() + 1e-6)
        target = target / (target.abs().max() + 1e-6)
        enrollment = enrollment / (enrollment.abs().max() + 1e-6)

        return {
            "mixture": mixture,
            "target": target,
            "enrollment": enrollment,
        }


class Libri2MixTargetSpeechDataset(Dataset):
    def __init__(
        self,
        metadata_csv: str | Path,
        librispeech_root: str | Path,
        wham_root: str | Path | None,
        sample_rate: int,
        mode: str = "min",
        mixture_type: str = "mix_both",
        segment_length: int = 48000,
        enrollment_length: int = 48000,
        target_source_strategy: str = "random",
        training: bool = True,
        seed: int = 42,
        max_items: int | None = None,
    ) -> None:
        self.metadata_csv = Path(metadata_csv)
        self.librispeech_root = Path(librispeech_root)
        self.wham_root = Path(wham_root) if wham_root is not None else None
        self.sample_rate = sample_rate
        self.mode = mode
        self.mixture_type = mixture_type
        self.segment_length = segment_length
        self.enrollment_length = enrollment_length
        self.target_source_strategy = target_source_strategy
        self.training = training
        self.seed = seed

        with open(self.metadata_csv, "r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            self.rows = list(reader)

        if not self.rows:
            raise ValueError(f"No rows found in metadata CSV: {self.metadata_csv}")

        if max_items is not None:
            self.rows = self.rows[:max_items]

        self.speaker_to_paths = defaultdict(list)
        for row in self.rows:
            for source_key in ("source_1_path", "source_2_path"):
                if source_key not in row or not row[source_key]:
                    continue
                source_path = row[source_key]
                speaker_id = self._speaker_id_from_path(source_path)
                self.speaker_to_paths[speaker_id].append(source_path)

        for speaker_id, paths in self.speaker_to_paths.items():
            self.speaker_to_paths[speaker_id] = sorted(set(paths))

        self.rows = self._filter_rows_with_alternate_enrollment(self.rows)
        if not self.rows:
            raise ValueError(
                "No Libri2Mix rows remain after enforcing same-speaker/different-utterance enrollment."
            )

    def __len__(self) -> int:
        return len(self.rows)

    def _rng(self, index: int) -> random.Random:
        return random.Random(self.seed + index)

    def _speaker_id_from_path(self, path_value: str) -> str:
        path = Path(path_value)
        if len(path.parts) >= 3:
            return path.parts[-3]

        stem_parts = path.stem.split("-")
        if stem_parts:
            return stem_parts[0]
        raise ValueError(f"Unable to infer speaker ID from path: {path_value}")

    def _resolve_audio_path(self, root: Path | None, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path

        candidate_paths = []
        if root is not None:
            candidate_paths.append(root / path)
        candidate_paths.append(self.metadata_csv.parent / path)
        candidate_paths.append(Path(path_value))

        for candidate in candidate_paths:
            if candidate.exists():
                return candidate

        return candidate_paths[0]

    def _load_librispeech(self, relative_path: str) -> np.ndarray:
        audio_path = self._resolve_audio_path(self.librispeech_root, relative_path)
        audio, sr = load_audio(audio_path)
        if sr != self.sample_rate:
            raise ValueError(
                f"Sample-rate mismatch for {audio_path}: expected {self.sample_rate}, got {sr}"
            )
        return audio

    def _load_noise(self, relative_path: str, target_length: int) -> np.ndarray:
        if self.mixture_type == "mix_clean":
            return np.zeros(target_length, dtype=np.float32)
        if self.wham_root is None:
            raise ValueError("wham_root must be provided when mixture_type uses noise.")

        noise_path = self._resolve_audio_path(self.wham_root, relative_path)
        noise, sr = load_audio(noise_path)
        if sr != self.sample_rate:
            raise ValueError(
                f"Sample-rate mismatch for {noise_path}: expected {self.sample_rate}, got {sr}"
            )
        if len(noise) < target_length:
            repeats = int(np.ceil(target_length / max(len(noise), 1)))
            noise = np.tile(noise, repeats)
        return noise[:target_length]

    def _select_target_index(self, rng: random.Random) -> int:
        if self.target_source_strategy == "source_1":
            return 1
        if self.target_source_strategy == "source_2":
            return 2
        if self.target_source_strategy == "random":
            return 1 if rng.random() < 0.5 else 2
        raise ValueError(f"Unsupported target_source_strategy: {self.target_source_strategy}")

    def _has_alternate_enrollment(self, current_source_path: str) -> bool:
        speaker_id = self._speaker_id_from_path(current_source_path)
        return any(path != current_source_path for path in self.speaker_to_paths[speaker_id])

    def _filter_rows_with_alternate_enrollment(self, rows: list[Dict[str, str]]) -> list[Dict[str, str]]:
        filtered_rows = []
        for row in rows:
            source_1_ok = self._has_alternate_enrollment(row["source_1_path"])
            source_2_ok = self._has_alternate_enrollment(row["source_2_path"])

            if self.target_source_strategy == "source_1" and source_1_ok:
                filtered_rows.append(row)
            elif self.target_source_strategy == "source_2" and source_2_ok:
                filtered_rows.append(row)
            elif self.target_source_strategy == "random" and (source_1_ok or source_2_ok):
                filtered_rows.append(row)
        return filtered_rows

    def _select_target_index_for_row(self, row: Dict[str, str], rng: random.Random) -> int:
        source_1_ok = self._has_alternate_enrollment(row["source_1_path"])
        source_2_ok = self._has_alternate_enrollment(row["source_2_path"])

        if self.target_source_strategy == "source_1":
            if not source_1_ok:
                raise ValueError("source_1 target has no alternate enrollment utterance.")
            return 1
        if self.target_source_strategy == "source_2":
            if not source_2_ok:
                raise ValueError("source_2 target has no alternate enrollment utterance.")
            return 2
        if self.target_source_strategy == "random":
            candidates = []
            if source_1_ok:
                candidates.append(1)
            if source_2_ok:
                candidates.append(2)
            if not candidates:
                raise ValueError("No valid target source has an alternate enrollment utterance.")
            return candidates[rng.randrange(len(candidates))]
        raise ValueError(f"Unsupported target_source_strategy: {self.target_source_strategy}")

    def _select_enrollment_path(self, current_source_path: str, rng: random.Random) -> str:
        speaker_id = self._speaker_id_from_path(current_source_path)
        candidates = [path for path in self.speaker_to_paths[speaker_id] if path != current_source_path]
        if not candidates:
            raise ValueError(
                f"No alternate enrollment utterance found for speaker {speaker_id} and source {current_source_path}."
            )
        return candidates[rng.randrange(len(candidates))]

    def _sample_offset(self, total_length: int, clip_length: int, rng: random.Random) -> int:
        if total_length <= clip_length:
            return 0
        if self.training:
            return rng.randint(0, total_length - clip_length)
        return (total_length - clip_length) // 2

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows[index]
        rng = self._rng(index)

        if self.mixture_type == "mix_single":
            if not self._has_alternate_enrollment(row["source_1_path"]):
                raise ValueError("mix_single sample has no alternate enrollment utterance for source_1.")
            target_index = 1
            interferer_source = np.zeros(1, dtype=np.float32)
        else:
            target_index = self._select_target_index_for_row(row, rng)
            interferer_index = 1 if target_index == 2 else 2
            interferer_source = self._load_librispeech(row[f"source_{interferer_index}_path"])
            interferer_source = interferer_source * float(row[f"source_{interferer_index}_gain"])

        target_source = self._load_librispeech(row[f"source_{target_index}_path"])
        target_source = target_source * float(row[f"source_{target_index}_gain"])

        base_sources = [target_source]
        if self.mixture_type != "mix_single":
            base_sources.append(interferer_source)

        noise_length = max(len(source) for source in base_sources)
        if self.mixture_type in {"mix_both", "mix_single"}:
            noise = self._load_noise(row["noise_path"], noise_length) * float(row["noise_gain"])
            base_sources.append(noise)

        sources = fit_audio_lengths(base_sources, self.mode)
        target = sources[0]
        mixture = np.sum(np.stack(sources, axis=0), axis=0)

        segment_offset = self._sample_offset(len(mixture), self.segment_length, rng)
        mixture = crop_or_pad(mixture, self.segment_length, offset=segment_offset)
        target = crop_or_pad(target, self.segment_length, offset=segment_offset)

        enrollment_path = self._select_enrollment_path(row[f"source_{target_index}_path"], rng)
        enrollment = self._load_librispeech(enrollment_path)
        enroll_offset = self._sample_offset(len(enrollment), self.enrollment_length, rng)
        enrollment = crop_or_pad(enrollment, self.enrollment_length, offset=enroll_offset)

        return {
            "mixture": to_tensor(mixture),
            "target": to_tensor(target),
            "enrollment": to_tensor(enrollment),
        }
