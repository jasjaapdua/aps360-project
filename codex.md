# Codex Prompt --- Song Lyrics Generation Deep Learning Project

## Situation

You are building a **Fundamentals of Deep Learning project** that
generates song lyrics using both a **baseline n-gram language model**
and a **neural model (LSTM, optionally Transformer)** trained on lyric
datasets. The code should be clean, modular, and suitable for a course
project, with clear documentation and reproducibility.

## Task

Produce a complete project repository with:

-   clean architecture
-   modular Python code
-   PyTorch training pipeline
-   baseline + deep learning model
-   dataset preprocessing
-   lyric generation utilities
-   evaluation scripts
-   documentation and notebooks

The result should resemble a **professional ML research project
repository**.

------------------------------------------------------------------------

# Codex Prompt

You are a senior machine learning engineer.\
Create a **complete, production-quality deep learning project
repository** for generating song lyrics using PyTorch.

The project should be written in **Python 3.11+**, structured like a
professional ML research repo, and heavily documented so that a student
can understand every part of the pipeline.

The repository must implement:

1.  A **baseline N-gram language model**
2.  An **LSTM-based neural lyric generator**
3.  Optional **Transformer-based generator**
4.  Dataset preprocessing pipeline
5.  Training and evaluation scripts
6.  Lyric generation utilities
7.  Documentation explaining every component

The project must prioritize:

-   modular architecture
-   clear docstrings
-   reproducibility
-   readability
-   separation of concerns

Use **PyTorch**, **HuggingFace datasets**, and standard ML tooling.

------------------------------------------------------------------------

# Repository Structure

    lyrics-generation-project/
    │
    ├── README.md
    ├── requirements.txt
    ├── setup.py
    ├── .gitignore
    │
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   └── dataset_info.md
    │
    ├── configs/
    │   ├── training_config.yaml
    │   ├── model_config.yaml
    │   └── dataset_config.yaml
    │
    ├── notebooks/
    │   ├── exploratory_data_analysis.ipynb
    │   ├── baseline_ngram_demo.ipynb
    │   └── lyrics_generation_demo.ipynb
    │
    ├── src/
    │   ├── __init__.py
    │   │
    │   ├── data/
    │   │   ├── dataset_loader.py
    │   │   ├── preprocessing.py
    │   │   ├── tokenizer.py
    │   │   └── dataset.py
    │   │
    │   ├── models/
    │   │   ├── ngram_model.py
    │   │   ├── lstm_model.py
    │   │   └── transformer_model.py
    │   │
    │   ├── training/
    │   │   ├── train_lstm.py
    │   │   ├── train_transformer.py
    │   │   └── trainer.py
    │   │
    │   ├── generation/
    │   │   └── generate_lyrics.py
    │   │
    │   ├── evaluation/
    │   │   ├── metrics.py
    │   │   └── evaluate_models.py
    │   │
    │   ├── utils/
    │   │   ├── text_utils.py
    │   │   ├── file_utils.py
    │   │   └── seed.py
    │   │
    │   └── visualization/
    │       └── training_plots.py
    │
    └── scripts/
        ├── download_datasets.py
        ├── preprocess_data.py
        ├── train_baseline.py
        ├── train_lstm.py
        └── generate_song.py

------------------------------------------------------------------------

# Documentation Requirements

Every module must include:

### Module docstring

``` python
"""
lstm_model.py

Defines the LSTM-based language model used for lyric generation.

This model learns to predict the next token in a lyric sequence.
Architecture:

Embedding -> LSTM -> Linear -> Softmax

Used during training and generation.
"""
```

### Function docstrings

Use **Google-style docstrings**.

``` python
def clean_lyrics(text: str) -> str:
    """
    Clean raw lyric text.

    Removes section markers, punctuation noise, and normalizes whitespace.

    Args:
        text: Raw lyric string.

    Returns:
        Cleaned lyric string.
    """
```

------------------------------------------------------------------------

# Dataset Pipeline

Implement preprocessing for lyric datasets.

Steps:

1.  Load lyrics dataset
2.  Filter English songs
3.  Remove short or empty songs
4.  Remove markup like `[chorus]`
5.  Normalize casing
6.  Tokenize text
7.  Build vocabulary
8.  Convert text → token sequences
9.  Train/val/test split at **song level**

Functions to implement:

-   `clean_lyrics()`
-   `build_vocabulary()`
-   `encode_sequence()`
-   `create_training_sequences()`
-   `split_dataset()`

------------------------------------------------------------------------

# Baseline Model (N-Gram)

Implement a trigram language model.

Features:

-   frequency-based probability estimation
-   Laplace smoothing
-   sampling-based text generation

File:

`src/models/ngram_model.py`

Functions:

-   `fit()`
-   `predict_next_word()`
-   `generate_text()`

------------------------------------------------------------------------

# LSTM Model

Implement a PyTorch LSTM language model.

Architecture:

    Embedding
    ↓
    2-layer LSTM
    ↓
    Dropout
    ↓
    Linear projection
    ↓
    Softmax over vocabulary

File:

`src/models/lstm_model.py`

Class:

`LyricsLSTM`

Methods:

-   `forward()`
-   `init_hidden()`

------------------------------------------------------------------------

# Transformer Model (Optional)

Implement a lightweight GPT-style transformer using:

`nn.TransformerEncoder`

File:

`src/models/transformer_model.py`

------------------------------------------------------------------------

# Training Pipeline

Create a reusable trainer class.

File:

`src/training/trainer.py`

Responsibilities:

-   training loop
-   validation loop
-   checkpointing
-   logging
-   early stopping

Methods:

-   `train_epoch()`
-   `validate_epoch()`
-   `save_checkpoint()`
-   `load_checkpoint()`

Loss:

`CrossEntropyLoss`

Optimizer:

`Adam`

------------------------------------------------------------------------

# Generation

Create lyric generation utility.

File:

`src/generation/generate_lyrics.py`

Features:

-   temperature sampling
-   top-k sampling
-   prompt conditioning

Example:

``` python
generate_lyrics(
    prompt="love is a fire",
    max_length=50,
    temperature=0.8
)
```

------------------------------------------------------------------------

# Evaluation

File:

`src/evaluation/metrics.py`

Metrics:

-   perplexity
-   BLEU
-   distinct-n

------------------------------------------------------------------------

# Scripts

Download dataset:

    python scripts/download_datasets.py

Preprocess:

    python scripts/preprocess_data.py

Train baseline:

    python scripts/train_baseline.py

Train LSTM:

    python scripts/train_lstm.py

Generate lyrics:

    python scripts/generate_song.py

------------------------------------------------------------------------

# README Requirements

The README must include:

-   project overview
-   dataset description
-   model architecture
-   training instructions
-   example outputs
-   evaluation explanation

------------------------------------------------------------------------

# Code Quality Requirements

The generated code must:

-   follow **PEP8**
-   include **type hints**
-   include **docstrings**
-   separate data, models, and training logic
-   be easily extensible

Avoid monolithic scripts.

------------------------------------------------------------------------

# Final Requirement

Generate all code files so the project runs with:

    pip install -r requirements.txt

Training should be runnable immediately after installation.
