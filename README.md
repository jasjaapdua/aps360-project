# Lyrics Generation Project

Deep learning pipeline for generating song lyrics using both a baseline trigram language model and neural architectures (LSTM, optional Transformer). Designed for APS360 Fundamentals of Deep Learning.

## Features
- Dataset download & preprocessing using HuggingFace `datasets`
- Baseline trigram model with Laplace smoothing
- LSTM language model with configurable depth/width
- Optional Transformer language model
- Reusable training loop with checkpointing and early stopping
- Generation utilities with temperature / top-k sampling
- Evaluation: perplexity, BLEU, distinct-n diversity
- Reproducible configs and lightweight notebooks

## Project Structure
```
├── configs/                # YAML configs for data, model, training
├── data/                   # raw/processed storage (created on demand)
├── notebooks/              # EDA, baseline, generation demos
├── scripts/                # CLI workflows (download, preprocess, train, generate)
├── src/                    # Library code (data, models, training, evaluation, utils)
└── requirements.txt
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage
1) Download dataset (configurable in `configs/dataset_config.yaml`):
```bash
python scripts/download_datasets.py
```
2) Preprocess (clean, tokenize, split, encode, build vocab):
```bash
python scripts/preprocess_data.py
```
3) Train baseline trigram:
```bash
python scripts/train_baseline.py
```
4) Train LSTM model:
```bash
python scripts/train_lstm.py
```
(Optional) Train Transformer:
```bash
python scripts/train_transformer.py
```
5) Generate lyrics from a checkpoint:
```bash
python scripts/generate_song.py --prompt "love is a fire" --checkpoint checkpoints/model_best_epoch1.pt
```

## Configuration
- `configs/dataset_config.yaml`: dataset name/columns, vocab + split sizes
- `configs/model_config.yaml`: vocab size, embedding/hidden sizes, transformer params
- `configs/training_config.yaml`: optimizer, lr, batch size, epochs, early stopping

## Evaluation
Use `src/evaluation/metrics.py` utilities to compute perplexity, BLEU, and distinct-n on generated samples. See `notebooks/lyrics_generation_demo.ipynb` for examples.

## Example Output
```
prompt: love is a fire
model: lstm (temperature=0.8)
-> love is a fire that burns in the rain
```

## Notes
- Default dataset placeholder: `spotify/podbaby-lyrics`. Swap to any HF dataset with a `lyrics` column.
- Processed artifacts live under `data/processed/`.
