# Lyrics Generation Project

This repository contains a targeted deep learning project for lyric generation using PyTorch. The project includes an end-to-end pipeline for dataset ingestion, cleaning, tokenization, sequence dataset construction, model training, baseline comparison, generation, evaluation, and reporting.

## Project Overview

The core of this project is a language model for generating song lyrics. It includes:

- A custom **PyTorch LSTM language model** for next-token prediction
- A lightweight **word-level tokenizer** and vocabulary builder
- A **baseline n-gram language model** for performance comparison
- Full **data pipeline** with dataset loading, preprocessing, splitting, and dataset creation
- **Evaluation metrics** including loss, perplexity, and accuracy
- **Qualitative output generation** with prompt-based samples
- A CLI for training, evaluation, and generation

## Key Features

- Support for dataset loading from **Hugging Face** and **Kaggle**
- Configurable training and generation via `.env`
- Deterministic song-level **train/validation/test splits**
- **Model checkpointing** and tokenizer persistence
- Baseline and neural network comparison
- **Visualization** of training loss
- External unseen-data evaluation support

## Repository Structure

- `main.py` — CLI entrypoint with commands for training, baseline, evaluation, and generation
- `config.py` — centralized environment-driven configuration loader
- `data/loader.py` — dataset loader for Hugging Face and Kaggle sources
- `preprocessing/text_cleaner.py` — lyric cleaning utilities
- `preprocessing/tokenizer.py` — vocabulary construction and tokenization
- `project_datasets/sequence_dataset.py` — sequence dataset for next-token prediction
- `project_datasets/splitter.py` — train/validation/test splitting helpers
- `models/lstm_model.py` — PyTorch LSTM language model implementation
- `models/ngram_model.py` — n-gram baseline implementation
- `training/trainer.py` — training loop and optimization
- `training/generate.py` — sequence generation from the trained model
- `training/evaluation.py` — evaluation metrics and qualitative sample generation
- `training/persistence.py` — model save/load utilities
- `visualisation/plotter.py` — training loss plotting
- `requirements.txt` — Python package dependencies

## Installation

1. Clone the repository:

```bash
git clone <repo-url> aps360-project
cd aps360-project
```

2. Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your dataset configuration:

```ini
DATASET_SOURCE=huggingface
DATASET_ID=<your-hf-dataset-id>
HF_TOKEN=<your-hf-token-if-needed>
SEQ_LENGTH=20
BATCH_SIZE=64
EPOCHS=20
LEARNING_RATE=0.001
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1
DEVICE=cpu
```

> If using Kaggle, set `DATASET_SOURCE=kaggle` and `DATASET_ID=<kaggle-dataset-slug>`.

## Usage

### Training the LSTM model

```bash
python main.py train
```

This executes the pipeline steps:
- load dataset
- clean lyrics
- split data
- build tokenizer
- encode data
- build sequence dataset
- train LSTM model

### Training the baseline n-gram model

```bash
python main.py baseline
```

### Evaluating the project end-to-end

```bash
python main.py evaluate
```

This command will run training and evaluation for both the baseline and the LSTM model, then produce evaluation artifacts.

### Generating lyrics

```bash
python main.py generate "seed text here"
```

If no trained checkpoint exists, the pipeline will train the model first and then generate text.

### Running the full pipeline

```bash
python main.py run
```

This command runs the same core pipeline as training, but is offered for convenience as a full workflow.

## Configuration

This project uses `config.py` to load environment variables via `python-dotenv`. The following settings are supported:

- `DATASET_PATH` — local dataset path (default: `data/lyrics.txt`)
- `DATASET_SOURCE` — `huggingface` or `kaggle`
- `DATASET_ID` — dataset identifier for selected source
- `HF_TOKEN` — Hugging Face token for authenticated access
- `HF_MAX_SONGS` — max songs to load from Hugging Face
- `SEQ_LENGTH` — sequence length for next-token prediction
- `BATCH_SIZE` — batch size for training and evaluation
- `EPOCHS` — number of training epochs
- `LEARNING_RATE` — optimizer learning rate
- `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO` — data split ratios
- `DEVICE` — `cpu` or `cuda`

## Model Details

### LSTM language model

Implemented in `models/lstm_model.py`, the model includes:
- Embedding layer
- Multi-layer LSTM
- Fully-connected linear layer to output vocabulary logits

The model is trained with `torch.optim.Adam` and `nn.CrossEntropyLoss`.

### N-gram baseline

Implemented in `models/ngram_model.py`, the baseline:
- builds n-gram count statistics
- predicts next token using weighted sampling
- generates sequences from a seed

## Evaluation

Evaluation includes:

- LSTM evaluation on held-out sequences using loss, perplexity, and accuracy
- n-gram evaluation using Laplace-smoothed negative log-likelihood
- qualitative generation samples from held-out prompts
- JSON report saving and text output artifacts

## Artifacts

- Model checkpoints are saved to `checkpoints/lstm_model.pt`
- Tokenizer vocabulary is saved to `checkpoints/tokenizer_vocab.json`
- Training loss plots are saved to `plots/training_loss.png`
- Reports can be stored in the configured `REPORT_DIR`


## Future Improvements

- add validation-based early stopping
- support beam search or temperature-controlled generation
- extend tokenizer to subword or byte-pair encoding
- add unit tests and automated experiment logging

