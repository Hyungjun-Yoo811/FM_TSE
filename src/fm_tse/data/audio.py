from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - dependency is declared in requirements.txt
    sf = None


def normalize_audio(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = np.max(np.abs(audio)) + 1e-8
    return peak * audio / max_abs


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    if sf is None:
        raise ImportError("soundfile is required to load LibriMix/LibriSpeech audio.")

    audio, sample_rate = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio.astype(np.float32), sample_rate


def crop_or_pad(
    audio: np.ndarray,
    length: int,
    *,
    offset: int | None = None,
) -> np.ndarray:
    if len(audio) >= length:
        if offset is None:
            offset = 0
        offset = max(0, min(offset, len(audio) - length))
        return audio[offset:offset + length]

    padded = np.zeros(length, dtype=np.float32)
    padded[:len(audio)] = audio
    return padded


def fit_audio_lengths(audios: list[np.ndarray], mode: str) -> list[np.ndarray]:
    if mode not in {"min", "max"}:
        raise ValueError(f"Unsupported mode: {mode}")

    target_length = min(len(audio) for audio in audios) if mode == "min" else max(len(audio) for audio in audios)
    return [crop_or_pad(audio, target_length) for audio in audios]


def to_tensor(audio: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(audio.astype(np.float32))


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
