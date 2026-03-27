# Flow Matching Target Speech Extraction Baseline

This repository provides a simple PyTorch baseline for Target Speech Extraction (TSE) with a conditional Flow Matching objective.

## What This Baseline Does

- Supports:
  - synthetic data for smoke tests
  - Libri2Mix metadata + LibriSpeech/WHAM roots for a real-data pipeline
- Conditions on:
  - `mixture`: noisy mixture containing target + interferer
  - `enrollment`: clean reference utterance from the target speaker
- Converts waveforms into log-mel spectrograms before feeding the network
- Learns a flow field that maps Gaussian noise to the target mel spectrogram
- Runs mel-domain training and Euler ODE sampling

## Project Structure

```text
configs/
  synthetic/
    baseline.json
  librimix/
    baseline.json
scripts/
  train.py
  infer.py
src/fm_tse/
  data/
    audio.py
    datasets.py
    features.py
    pipeline.py
  models/
    flow_matching.py
    networks.py
  utils/
    config.py
    metrics.py
    visualization.py
requirements.txt
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/train.py --config configs/librimix/baseline.json
python scripts/infer.py --checkpoint outputs_librimix/checkpoints/latest.pt --config configs/librimix/baseline.json --split valid
```

## Baseline Formulation

Given:

- target mel spectrogram `y`
- initial noise `z`
- interpolation time `t in [0, 1]`

the linear probability path is:

`x_t = (1 - t) * z + t * y`

The target vector field is:

`u_t = y - z`

The model predicts `v_theta(x_t, t, mixture, enrollment)` and is trained with:

`L = ||v_theta - u_t||^2`

At inference, we start from Gaussian noise and integrate the learned ODE from `t=0` to `t=1` in mel space.

## Notes

- This is intentionally lightweight and research-friendly, not production-optimized.
- The synthetic dataset is useful for code validation and ablations.
- For Libri2Mix, the pipeline reads the metadata CSV and reconstructs the mixture on the fly from:
  - LibriSpeech utterances
  - WHAM noise
  - gains stored in the metadata
- This means you do not need to pre-generate the large LibriMix waveform tree to start wiring the experiment code.
- Training writes plot artifacts such as loss curves, flow trajectory plots, and mel spectrogram panels under `output_dir/plots`.
- Inference writes mel panels and serialized mel tensors under `output_dir/samples`.
- Reported metrics now include mel-domain proxies (`mel_l1`, `spectral_convergence`, `frame_cosine_similarity`) and common TSE/separation scores reconstructed to waveform space (`SI-SDR`, `SI-SDRi`, `SI-SNR`, `SI-SNRi`, `SNR`, `PESQ`).
- The default script configuration now targets Libri2Mix TSE runs; use the synthetic config only for smoke tests or quick debugging.

## Libri2Mix Pipeline

Use [baseline.json](C:\Users\hjyoo\FM_TSE\configs\librimix\baseline.json) after your data download paths are ready.

- `metadata_csv`: Libri2Mix metadata file
- `librispeech_root`: root that contains split folders like `train-clean-100/...`
- `wham_root`: WHAM noise root that contains entries like `tr/...`, `cv/...`, `tt/...`
- `mode`: `min` or `max`
- `mixture_type`: `mix_clean` or `mix_both`

Example:

```bash
python scripts/train.py --config configs/librimix/baseline.json
python scripts/infer.py --checkpoint outputs_librimix/checkpoints/latest.pt --config configs/librimix/baseline.json --split valid
```
"# FM_TSE" 
