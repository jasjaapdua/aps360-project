"""Command-line entrypoint for lyric model training and generation.

This version is config-first:
- values are loaded from a singleton :class:`AppConfig` backed by `.env`,
- CLI accepts only lightweight overrides for convenience,
- most training/generation settings come from Python config dataclasses.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

from config import ConfigSingleton
from models import (
    TrainConfig,
    evaluate_trigram_accuracy,
    generate_text,
    generate_trigram,
    load_checkpoint,
    save_checkpoint,
    train_lstm_model,
    train_trigram,
)
from preprocessing import (
    LyricsTokenizer,
    build_vocabulary,
    load_huggingface_corpus,
    load_kaggle_corpus,
    load_local_corpus,
    load_vocabulary,
    save_vocabulary,
    split_corpus,
)
from utils import RunLogger, build_run_logger


SAMPLE_LYRICS = """midnight city lights are glowing in the rain
hold my hand and dance until the morning came
we were young and loud and burning like a flame
say my name say my name say my name

i hear the crowd singing louder than the drums
your heartbeat racing faster while the chorus comes
broken dreams and golden streets and restless runs
we sing on we sing on until the night is done

if the sky falls down we will rise again
every verse we write will wash away the pain
through the dark we find a small electric sun
we sing on we sing on until the night is done
"""


def set_seed(seed: int = 42) -> None:
    """Set Python and Torch random seeds for repeatable runs."""
    random.seed(seed)
    torch.manual_seed(seed)


def ensure_sample_dataset(path: Path) -> None:
    """Create a small built-in sample dataset when no file exists.

    Args:
        path: Target dataset path.
    """
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SAMPLE_LYRICS, encoding="utf-8")


def load_training_corpus(cfg) -> tuple[list[list[str]], str]:
    """Load corpus from the configured data source.

    Args:
        cfg: Application config object.

    Returns:
        Tuple `(corpus, source_info)`.
    """
    data_cfg = cfg.data

    if data_cfg.data_source == "local":
        if data_cfg.use_sample_if_missing:
            ensure_sample_dataset(Path(cfg.paths.local_input))
        return load_local_corpus(cfg.paths.local_input)

    if data_cfg.data_source == "huggingface":
        if not data_cfg.hf_dataset:
            raise ValueError("LYRICS_HF_DATASET must be set for huggingface data source.")
        return load_huggingface_corpus(
            dataset_name=data_cfg.hf_dataset,
            config_name=data_cfg.hf_config,
            split=data_cfg.hf_split,
            hf_token=data_cfg.hf_token,
            text_column=data_cfg.text_column,
            max_samples=data_cfg.max_samples,
        )

    if data_cfg.data_source == "kaggle":
        if not data_cfg.kaggle_dataset:
            raise ValueError("LYRICS_KAGGLE_DATASET must be set for kaggle data source.")
        return load_kaggle_corpus(
            dataset_ref=data_cfg.kaggle_dataset,
            text_column=data_cfg.text_column,
            file_pattern=data_cfg.kaggle_file_pattern,
            max_samples=data_cfg.max_samples,
        )

    raise ValueError(f"Unsupported data source: {data_cfg.data_source}")


def _safe_perplexity(loss_value: float) -> float:
    """Compute perplexity from cross-entropy loss with overflow protection."""
    return float("inf") if loss_value > 50 else math.exp(loss_value)


def _resolve_device(force_cpu: bool) -> str:
    """Choose runtime device based on config and CUDA availability."""
    return "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"


def _apply_train_overrides(cfg, args: argparse.Namespace):
    """Apply optional CLI overrides on top of base config.

    Args:
        cfg: Base app config loaded from singleton.
        args: Parsed CLI args containing optional override values.

    Returns:
        New app config with selected fields overridden.
    """
    data_cfg = cfg.data
    train_cfg = cfg.train

    if args.data_source is not None:
        data_cfg = replace(data_cfg, data_source=args.data_source)
    if args.input is not None:
        paths_cfg = replace(cfg.paths, local_input=args.input)
    else:
        paths_cfg = cfg.paths
    if args.text_column is not None:
        data_cfg = replace(data_cfg, text_column=args.text_column)
    if args.max_samples is not None:
        data_cfg = replace(data_cfg, max_samples=args.max_samples)
    if args.val_ratio is not None:
        data_cfg = replace(data_cfg, val_ratio=args.val_ratio)
    if args.preview_prompt is not None:
        train_cfg = replace(train_cfg, preview_prompt=args.preview_prompt)
    if args.cpu is not None:
        train_cfg = replace(train_cfg, cpu=args.cpu)

    if args.hf_dataset is not None:
        data_cfg = replace(data_cfg, hf_dataset=args.hf_dataset)
    if args.hf_config is not None:
        data_cfg = replace(data_cfg, hf_config=args.hf_config)
    if args.hf_split is not None:
        data_cfg = replace(data_cfg, hf_split=args.hf_split)

    if args.kaggle_dataset is not None:
        data_cfg = replace(data_cfg, kaggle_dataset=args.kaggle_dataset)
    if args.kaggle_file_pattern is not None:
        data_cfg = replace(data_cfg, kaggle_file_pattern=args.kaggle_file_pattern)

    return replace(cfg, paths=paths_cfg, data=data_cfg, train=train_cfg)


def _apply_generate_overrides(cfg, args: argparse.Namespace):
    """Apply optional generation CLI overrides on top of base config."""
    gen_cfg = cfg.generation
    paths_cfg = cfg.paths

    if args.prompt is not None:
        gen_cfg = replace(gen_cfg, prompt=args.prompt)
    if args.max_new_tokens is not None:
        gen_cfg = replace(gen_cfg, max_new_tokens=args.max_new_tokens)
    if args.temperature is not None:
        gen_cfg = replace(gen_cfg, temperature=args.temperature)
    if args.top_k is not None:
        gen_cfg = replace(gen_cfg, top_k=args.top_k)
    if args.cpu is not None:
        gen_cfg = replace(gen_cfg, cpu=args.cpu)

    if args.model is not None:
        paths_cfg = replace(paths_cfg, model_out=args.model)
    if args.vocab is not None:
        paths_cfg = replace(paths_cfg, vocab_out=args.vocab)

    return replace(cfg, paths=paths_cfg, generation=gen_cfg)


def _print_effective_config(cfg) -> None:
    """Print key effective config values for debugging/runtime validation."""
    print("Effective config:")
    print(f"  data_source={cfg.data.data_source}")
    print(f"  local_input={cfg.paths.local_input}")
    print(f"  model_out={cfg.paths.model_out}")
    print(f"  vocab_out={cfg.paths.vocab_out}")
    print(f"  text_column={cfg.data.text_column}")
    print(f"  val_ratio={cfg.data.val_ratio}")
    print(f"  log_file={cfg.logging.log_file}")


def train_command(args: argparse.Namespace) -> None:
    """Execute training using singleton config plus optional CLI overrides."""
    logger: RunLogger = args._run_logger
    cfg = args._base_cfg
    cfg = _apply_train_overrides(cfg, args)
    logger.info("effective_train_config", config=asdict(cfg))

    if args.print_config:
        _print_effective_config(cfg)

    set_seed(cfg.train.seed)
    corpus, source_info = load_training_corpus(cfg)
    logger.info("dataset_loaded", source_info=source_info, total_lines=len(corpus))

    if not corpus:
        raise ValueError("No usable lyric lines found after cleaning.")

    train_corpus, val_corpus = split_corpus(
        corpus,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.train.seed,
    )
    logger.info(
        "dataset_split",
        train_lines=len(train_corpus),
        validation_lines=len(val_corpus),
        val_ratio=cfg.data.val_ratio,
    )
    stoi, _ = build_vocabulary(
        train_corpus,
        min_freq=cfg.train.min_freq,
        max_size=cfg.train.max_vocab,
    )
    tokenizer = LyricsTokenizer(stoi)

    train_config = TrainConfig(
        seq_len=cfg.train.seq_len,
        batch_size=cfg.train.batch_size,
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        embedding_dim=cfg.train.embedding_dim,
        hidden_dim=cfg.train.hidden_dim,
        num_layers=cfg.train.num_layers,
        dropout=cfg.train.dropout,
    )

    device = _resolve_device(force_cpu=cfg.train.cpu)
    model, train_losses, val_losses = train_lstm_model(
        train_corpus=train_corpus,
        tokenizer=tokenizer,
        config=train_config,
        val_corpus=val_corpus,
        device=device,
    )
    logger.info(
        "training_complete",
        device=device,
        train_loss=train_losses[-1],
        train_perplexity=_safe_perplexity(train_losses[-1]),
        val_loss=val_losses[-1] if val_losses else None,
        val_perplexity=_safe_perplexity(val_losses[-1]) if val_losses else None,
    )

    save_vocabulary(stoi, cfg.paths.vocab_out)
    Path(cfg.paths.model_out).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, cfg.paths.model_out, train_config)
    logger.info("artifacts_saved", model_out=cfg.paths.model_out, vocab_out=cfg.paths.vocab_out)

    trigram_model = train_trigram(train_corpus)

    print(f"Source: {source_info}")
    print(f"Total lines: {len(corpus)}")
    print(f"Train lines: {len(train_corpus)} | Validation lines: {len(val_corpus)}")
    print(f"Vocab size (train only): {len(stoi)}")
    print(f"Device: {device}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final train perplexity: {_safe_perplexity(train_losses[-1]):.2f}")

    if val_losses:
        print(f"Final val loss: {val_losses[-1]:.4f}")
        print(f"Final val perplexity: {_safe_perplexity(val_losses[-1]):.2f}")

    if val_corpus:
        trigram_acc = evaluate_trigram_accuracy(trigram_model, val_corpus)
        print(f"Trigram val next-token accuracy: {trigram_acc * 100:.2f}%")
        logger.info("trigram_validation", next_token_accuracy=trigram_acc)

    print("\nLSTM sample:")
    print(
        generate_text(
            model,
            tokenizer,
            prompt=cfg.train.preview_prompt,
            max_new_tokens=30,
            device=device,
        )
    )
    print("\nTrigram sample:")
    print(generate_trigram(trigram_model, prompt=cfg.train.preview_prompt, max_tokens=30))


def generate_command(args: argparse.Namespace) -> None:
    """Execute text generation using singleton config plus optional overrides."""
    logger: RunLogger = args._run_logger
    cfg = args._base_cfg
    cfg = _apply_generate_overrides(cfg, args)
    logger.info("effective_generate_config", config=asdict(cfg))

    if args.print_config:
        _print_effective_config(cfg)

    stoi = load_vocabulary(cfg.paths.vocab_out)
    tokenizer = LyricsTokenizer(stoi)

    device = _resolve_device(force_cpu=cfg.generation.cpu)
    model, _ = load_checkpoint(cfg.paths.model_out, device=device)

    text = generate_text(
        model,
        tokenizer,
        prompt=cfg.generation.prompt,
        max_new_tokens=cfg.generation.max_new_tokens,
        temperature=cfg.generation.temperature,
        top_k=cfg.generation.top_k,
        device=device,
    )
    logger.info(
        "generation_complete",
        device=device,
        prompt=cfg.generation.prompt,
        output_token_estimate=len(text.split()),
        output_characters=len(text),
    )
    print(text)


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI parser with minimal, optional override arguments."""
    parser = argparse.ArgumentParser(description="Simple lyric generation project")

    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train LSTM lyric model")
    train.add_argument("--env-file", type=str, default=".env", help="Path to environment file")
    train.add_argument("--data-source", choices=["local", "huggingface", "kaggle"], default=None)
    train.add_argument("--input", type=str, default=None)

    train.add_argument("--hf-dataset", type=str, default=None)
    train.add_argument("--hf-config", type=str, default=None)
    train.add_argument("--hf-split", type=str, default=None)

    train.add_argument("--kaggle-dataset", type=str, default=None)
    train.add_argument("--kaggle-file-pattern", type=str, default=None)

    train.add_argument("--text-column", type=str, default=None)
    train.add_argument("--max-samples", type=int, default=None)
    train.add_argument("--val-ratio", type=float, default=None)

    train.add_argument("--preview-prompt", type=str, default=None)
    train.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=None)
    train.add_argument("--print-config", action="store_true")
    train.set_defaults(func=train_command)

    gen = sub.add_parser("generate", help="Generate lyrics from trained model")
    gen.add_argument("--env-file", type=str, default=".env", help="Path to environment file")
    gen.add_argument("--model", type=str, default=None)
    gen.add_argument("--vocab", type=str, default=None)
    gen.add_argument("--prompt", type=str, default=None)
    gen.add_argument("--max-new-tokens", type=int, default=None)
    gen.add_argument("--temperature", type=float, default=None)
    gen.add_argument("--top-k", type=int, default=None)
    gen.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=None)
    gen.add_argument("--print-config", action="store_true")
    gen.set_defaults(func=generate_command)

    return parser


def main() -> None:
    """Program entrypoint used by `python main.py ...`."""
    parser = build_parser()
    args = parser.parse_args()

    base_cfg = ConfigSingleton.get(env_file=args.env_file, force_reload=True)
    run_logger = build_run_logger(
        log_file=base_cfg.logging.log_file,
        level=base_cfg.logging.level,
    )
    run_logger.info(
        "command_invoked",
        command=args.command,
        argv=sys.argv[1:],
        env_file=args.env_file,
        base_config=asdict(base_cfg),
    )

    args._base_cfg = base_cfg
    args._run_logger = run_logger

    try:
        args.func(args)
        run_logger.finish(status="success", command=args.command)
    except Exception as exc:
        run_logger.error(
            "command_failed",
            command=args.command,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        run_logger.finish(status="error", command=args.command)
        raise


if __name__ == "__main__":
    main()
