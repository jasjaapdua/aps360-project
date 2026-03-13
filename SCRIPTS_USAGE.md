# Scripts Usage Guide

This document describes how to use all Python scripts/modules in this project, with runnable examples and sample `.env` configurations.

## 1) `main.py` (CLI entrypoint)

`main.py` is the script you run directly.

Commands:
- `train`
- `generate`

General form:

```bash
python main.py <command> [optional-overrides]
```

Use a specific env file:

```bash
python main.py train --env-file .env
python main.py generate --env-file .env
```

### 1.1 `train`

```bash
python main.py train
```

Optional overrides:

```bash
python main.py train \
  --data-source huggingface \
  --hf-dataset "neelshah18/song-lyrics-dataset" \
  --text-column text \
  --max-samples 50000 \
  --val-ratio 0.1
```

```bash
python main.py train \
  --data-source kaggle \
  --kaggle-dataset "username/dataset-name" \
  --kaggle-file-pattern "*.csv" \
  --text-column lyrics
```

```bash
python main.py train --input "data/*.txt" --print-config
```

### 1.2 `generate`

```bash
python main.py generate
```

Optional overrides:

```bash
python main.py generate --prompt "city lights" --temperature 0.8 --top-k 30
```

```bash
python main.py generate \
  --model artifacts/lyrics_lstm.pt \
  --vocab artifacts/vocab.json \
  --max-new-tokens 80
```

## 2) Example `.env` Configurations

Copy starter template:

```bash
cp .env.example .env
```

### 2.1 Local text files

```env
LYRICS_DATA_SOURCE=local
LYRICS_LOCAL_INPUT=data/sample_lyrics.txt
LYRICS_USE_SAMPLE_IF_MISSING=true

LYRICS_TEXT_COLUMN=text
LYRICS_MAX_SAMPLES=
LYRICS_VAL_RATIO=0.1

LYRICS_MODEL_OUT=artifacts/lyrics_lstm.pt
LYRICS_VOCAB_OUT=artifacts/vocab.json

LYRICS_LOG_FILE=logs/cli_runs.log
LYRICS_LOG_LEVEL=INFO
```

Run:

```bash
python main.py train
python main.py generate
```

### 2.2 Hugging Face dataset

```env
LYRICS_DATA_SOURCE=huggingface
LYRICS_HF_DATASET=neelshah18/song-lyrics-dataset
LYRICS_HF_CONFIG=
LYRICS_HF_SPLIT=train

LYRICS_TEXT_COLUMN=text
LYRICS_MAX_SAMPLES=50000
LYRICS_VAL_RATIO=0.1

LYRICS_MODEL_OUT=artifacts/hf_lyrics_lstm.pt
LYRICS_VOCAB_OUT=artifacts/hf_vocab.json

LYRICS_LOG_FILE=logs/hf_runs.log
LYRICS_LOG_LEVEL=INFO
```

Run:

```bash
python main.py train
python main.py generate --model artifacts/hf_lyrics_lstm.pt --vocab artifacts/hf_vocab.json
```

### 2.3 Kaggle dataset

```env
LYRICS_DATA_SOURCE=kaggle
LYRICS_KAGGLE_DATASET=username/dataset-name
LYRICS_KAGGLE_FILE_PATTERN=*.csv

LYRICS_TEXT_COLUMN=lyrics
LYRICS_MAX_SAMPLES=80000
LYRICS_VAL_RATIO=0.1

LYRICS_MODEL_OUT=artifacts/kg_lyrics_lstm.pt
LYRICS_VOCAB_OUT=artifacts/kg_vocab.json

LYRICS_LOG_FILE=logs/kaggle_runs.log
LYRICS_LOG_LEVEL=INFO
```

Run:

```bash
python main.py train
python main.py generate --model artifacts/kg_lyrics_lstm.pt --vocab artifacts/kg_vocab.json
```

## 3) Environment Variables Explained

All variables are read in `config/app_config.py`. Empty values are treated as unset for optional fields.

### 3.1 Data source and ingestion

- `LYRICS_DATA_SOURCE` (string, default: `local`)
  - Selects dataset backend: `local`, `huggingface`, or `kaggle`.
- `LYRICS_LOCAL_INPUT` (string path/glob, default: `data/sample_lyrics.txt`)
  - Local file path, directory, or glob used when data source is `local`.
- `LYRICS_USE_SAMPLE_IF_MISSING` (bool, default: `true`)
  - If true and local input is missing, creates a built-in sample lyric file.
- `LYRICS_HF_DATASET` (string, default: unset)
  - Hugging Face dataset id (required when source is `huggingface`).
- `LYRICS_HF_CONFIG` (string, default: unset)
  - Optional HF config/subset name.
- `LYRICS_HF_SPLIT` (string, default: `train`)
  - HF split to load.
- `LYRICS_KAGGLE_DATASET` (string, default: unset)
  - Kaggle dataset reference, e.g. `user/dataset` (required for `kaggle` source).
- `LYRICS_KAGGLE_FILE_PATTERN` (string glob, default: `*.csv`)
  - File pattern searched in downloaded Kaggle dataset.
- `LYRICS_TEXT_COLUMN` (string, default: `text`)
  - Column containing lyric text for HF/Kaggle tabular data.
- `LYRICS_MAX_SAMPLES` (int, default: unset)
  - Optional cap on number of rows/samples consumed from remote datasets.
- `LYRICS_VAL_RATIO` (float, default: `0.1`)
  - Fraction of corpus used for validation split.

### 3.2 Artifact output paths

- `LYRICS_MODEL_OUT` (string path, default: `artifacts/lyrics_lstm.pt`)
  - Checkpoint path for saved LSTM model.
- `LYRICS_VOCAB_OUT` (string path, default: `artifacts/vocab.json`)
  - JSON path for saved vocabulary mapping.

### 3.3 Logging

- `LYRICS_LOG_FILE` (string path, default: `logs/cli_runs.log`)
  - JSONL file that receives CLI run logs.
- `LYRICS_LOG_LEVEL` (string, default: `INFO`)
  - Logging level for run events (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).

### 3.4 Training hyperparameters

- `LYRICS_MIN_FREQ` (int, default: `1`)
  - Minimum token frequency to include in vocab.
- `LYRICS_MAX_VOCAB` (int, default: `5000`)
  - Maximum vocabulary size including special tokens.
- `LYRICS_SEQ_LEN` (int, default: `12`)
  - Sequence length for next-token training windows.
- `LYRICS_BATCH_SIZE` (int, default: `16`)
  - Batch size used for training and validation dataloaders.
- `LYRICS_EPOCHS` (int, default: `10`)
  - Number of training epochs.
- `LYRICS_LR` (float, default: `0.001`)
  - Adam optimizer learning rate.
- `LYRICS_EMBEDDING_DIM` (int, default: `128`)
  - Embedding size for token vectors.
- `LYRICS_HIDDEN_DIM` (int, default: `256`)
  - LSTM hidden state size.
- `LYRICS_NUM_LAYERS` (int, default: `2`)
  - Number of stacked LSTM layers.
- `LYRICS_DROPOUT` (float, default: `0.2`)
  - Dropout between recurrent layers (`num_layers > 1`).
- `LYRICS_SEED` (int, default: `42`)
  - Random seed for reproducibility (Python + Torch).
- `LYRICS_CPU` (bool, default: `false`)
  - Forces training on CPU when true.
- `LYRICS_PREVIEW_PROMPT` (string, default: `we sing`)
  - Prompt used to print sample output after training.

### 3.5 Generation parameters

- `LYRICS_PROMPT` (string, default: `midnight`)
  - Default prompt for `python main.py generate`.
- `LYRICS_MAX_NEW_TOKENS` (int, default: `40`)
  - Maximum number of generated tokens.
- `LYRICS_TEMPERATURE` (float, default: `0.9`)
  - Sampling temperature (lower = safer, higher = more random).
- `LYRICS_TOP_K` (int, default: `20`)
  - Restricts sampling to top-k token candidates per step.
- `LYRICS_GENERATE_CPU` (bool, default: `false`)
  - Forces generation on CPU when true.

## 4) Config Script/Module Usage

### `config/app_config.py`

Purpose:
- Loads `.env` + environment variables into typed dataclasses.
- Exposes singleton config via `ConfigSingleton.get(...)`.

Example:

```python
from config import ConfigSingleton

cfg = ConfigSingleton.get(env_file=".env")
print(cfg.data.data_source)
print(cfg.paths.model_out)
```

## 5) Preprocessing Modules

### `preprocessing/clean_lyrics.py`

Purpose:
- Clean text and build tokenized corpus from `.txt` files.

Example:

```python
from preprocessing.clean_lyrics import load_corpus

corpus = load_corpus(["data/sample_lyrics.txt"])
print(corpus[:2])
```

### `preprocessing/build_vocabulary.py`

Purpose:
- Build/save/load vocabulary maps.

Example:

```python
from preprocessing.build_vocabulary import build_vocabulary, save_vocabulary

corpus = [["we", "sing"], ["sing", "on"]]
stoi, itos = build_vocabulary(corpus, min_freq=1, max_size=100)
save_vocabulary(stoi, "artifacts/vocab.json")
```

### `preprocessing/tokenizer.py`

Purpose:
- Word-level tokenization + encode/decode.

Example:

```python
from preprocessing.tokenizer import LyricsTokenizer

stoi = {"<pad>":0, "<unk>":1, "<bos>":2, "<eos>":3, "we":4, "sing":5}
tok = LyricsTokenizer(stoi)
ids = tok.encode("we sing")
print(ids)
print(tok.decode(ids))
```

### `preprocessing/dataset_sources.py`

Purpose:
- Load corpus from local files, Hugging Face, or Kaggle.
- Split corpus into train/validation.

Examples:

```python
from preprocessing.dataset_sources import load_local_corpus, split_corpus

corpus, source = load_local_corpus("data/*.txt")
train_corpus, val_corpus = split_corpus(corpus, val_ratio=0.1, seed=42)
```

```python
from preprocessing.dataset_sources import load_huggingface_corpus

corpus, source = load_huggingface_corpus(
    dataset_name="neelshah18/song-lyrics-dataset",
    text_column="text",
    split="train",
    max_samples=1000,
)
```

## 6) Model Modules

### `models/lstm.py`

Purpose:
- Train/evaluate/generate with LSTM language model.

Example:

```python
from models.lstm import TrainConfig, train_lstm_model, generate_text
from preprocessing.tokenizer import LyricsTokenizer

corpus = [["we", "sing"], ["sing", "on"]]
stoi = {"<pad>":0, "<unk>":1, "<bos>":2, "<eos>":3, "we":4, "sing":5, "on":6}
tok = LyricsTokenizer(stoi)
config = TrainConfig(seq_len=2, batch_size=2, epochs=1)
model, train_losses, val_losses = train_lstm_model(corpus, tok, config, val_corpus=[])
print(generate_text(model, tok, prompt="we", max_new_tokens=10))
```

### `models/trigram.py`

Purpose:
- Train/generate/evaluate trigram baseline.

Example:

```python
from models.trigram import train_trigram, generate_trigram, evaluate_trigram_accuracy

corpus = [["we", "sing"], ["we", "sing", "on"]]
model = train_trigram(corpus)
print(generate_trigram(model, prompt="we"))
print(evaluate_trigram_accuracy(model, corpus))
```

## 7) Logging Module

### `utils/run_logger.py`

Purpose:
- Writes structured JSON-line logs for each CLI invocation.

Example:

```python
from utils.run_logger import build_run_logger

logger = build_run_logger(log_file="logs/manual.log", level="INFO")
logger.info("custom_event", detail="hello")
logger.finish(status="success")
```

Log file default (from env):
- `logs/cli_runs.log`

## 8) Package `__init__` modules

These files expose public APIs and are not executed directly:
- `config/__init__.py`
- `preprocessing/__init__.py`
- `models/__init__.py`
- `utils/__init__.py`

Use them for cleaner imports:

```python
from config import ConfigSingleton
from models import train_trigram
from preprocessing import LyricsTokenizer
```

## 9) Quick Run Checklist

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create `.env`:

```bash
cp .env.example .env
```

3. Train:

```bash
python main.py train
```

4. Generate:

```bash
python main.py generate
```
