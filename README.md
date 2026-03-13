# Lyric Generation with Deep Learning

A compact project for generating song lyrics using:
- a word-level LSTM language model, and
- a trigram baseline.

The project is now **config-first**:
- runtime settings are loaded from a `.env` file,
- Python dataclass config provides defaults and validation,
- config access is managed through a singleton.
- every CLI command writes structured run logs to a log file.

This keeps CLI usage short while still allowing targeted overrides.

## What Changed

- Added a singleton config layer in `config/app_config.py`
- Added `.env.example` with all supported settings
- Refactored CLI to rely on config by default
- Reduced required CLI arguments for `train` and `generate`
- Kept optional CLI overrides for quick experiments

## Project Structure

```text
aps360-project/
├── config/
│   ├── __init__.py
│   └── app_config.py
├── data/
│   └── sample_lyrics.txt
├── models/
│   ├── __init__.py
│   ├── lstm.py
│   └── trigram.py
├── preprocessing/
│   ├── __init__.py
│   ├── build_vocabulary.py
│   ├── clean_lyrics.py
│   ├── dataset_sources.py
│   └── tokenizer.py
├── .env.example
├── main.py
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Configure with `.env`

1. Create your env file:

```bash
cp .env.example .env
```

2. Edit `.env` based on your data source.

## Minimal CLI Usage

Train with env-configured settings:

```bash
python main.py train
```

Generate with env-configured settings:

```bash
python main.py generate
```

Each command appends structured JSON-line events to `LYRICS_LOG_FILE`
(default: `logs/cli_runs.log`).

## Optional CLI Overrides

You can override a few settings without editing `.env`.

Train examples:

```bash
python main.py train --data-source huggingface --hf-dataset "neelshah18/song-lyrics-dataset" --text-column text
python main.py train --data-source kaggle --kaggle-dataset "username/dataset-name" --text-column lyrics
python main.py train --input "data/*.txt" --val-ratio 0.2
```

Generate examples:

```bash
python main.py generate --prompt "city lights" --temperature 0.8 --top-k 30
```

Use a different env file:

```bash
python main.py train --env-file .env.dev
python main.py generate --env-file .env.dev
```

Print effective runtime config:

```bash
python main.py train --print-config
python main.py generate --print-config
```

## Supported Environment Variables

### Data source selection
- `LYRICS_DATA_SOURCE`: `local`, `huggingface`, or `kaggle`

### Local dataset
- `LYRICS_LOCAL_INPUT`
- `LYRICS_USE_SAMPLE_IF_MISSING`

### Hugging Face dataset
- `LYRICS_HF_DATASET`
- `LYRICS_HF_CONFIG`
- `LYRICS_HF_SPLIT`

### Kaggle dataset
- `LYRICS_KAGGLE_DATASET`
- `LYRICS_KAGGLE_FILE_PATTERN`

### Shared dataset options
- `LYRICS_TEXT_COLUMN`
- `LYRICS_MAX_SAMPLES`
- `LYRICS_VAL_RATIO`

### Artifact paths
- `LYRICS_MODEL_OUT`
- `LYRICS_VOCAB_OUT`

### Logging
- `LYRICS_LOG_FILE`
- `LYRICS_LOG_LEVEL`

### Training config
- `LYRICS_MIN_FREQ`
- `LYRICS_MAX_VOCAB`
- `LYRICS_SEQ_LEN`
- `LYRICS_BATCH_SIZE`
- `LYRICS_EPOCHS`
- `LYRICS_LR`
- `LYRICS_EMBEDDING_DIM`
- `LYRICS_HIDDEN_DIM`
- `LYRICS_NUM_LAYERS`
- `LYRICS_DROPOUT`
- `LYRICS_SEED`
- `LYRICS_CPU`
- `LYRICS_PREVIEW_PROMPT`

### Generation config
- `LYRICS_PROMPT`
- `LYRICS_MAX_NEW_TOKENS`
- `LYRICS_TEMPERATURE`
- `LYRICS_TOP_K`
- `LYRICS_GENERATE_CPU`

## Singleton Config Design

`ConfigSingleton.get(env_file=".env")` returns a cached `AppConfig` instance.

- First call loads `.env` + process environment.
- Subsequent calls reuse the same object.
- Passing a different `env_file` reloads config.

The `AppConfig` tree is split into:
- `paths`
- `data`
- `train`
- `generation`
- `logging`

This keeps config organized and reduces argument plumbing.

## Logging System

The CLI now records run metadata to a file logger in `utils/run_logger.py`.

For every invocation, log events include:
- `run_started`
- `command_invoked` (argv, env file, base config)
- `effective_train_config` or `effective_generate_config`
- data/training/generation summary events
- `run_finished` (status + duration)

On failure, `command_failed` is also written with exception type/message.

Log format is one JSON object per line, making it easy to parse programmatically.

## Validation Metrics

Training reports:
- train loss / train perplexity
- validation loss / validation perplexity (if validation split exists)
- trigram validation next-token accuracy

## Troubleshooting

### Hugging Face import conflict

A local folder named `datasets/` can shadow Hugging Face `datasets`. The loader mitigates this, but renaming the local folder is still recommended if import issues persist.

### Kaggle access

For private/restricted datasets, ensure Kaggle authentication is configured for `kagglehub`.

### PyTorch shared-memory issue in restricted sandboxes

If you see errors like `OMP: Error #179: Can't open SHM2`, run training in a normal local environment.
