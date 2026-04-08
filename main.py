"""
main.py

Command line interface for the lyrics generation project.
"""
import argparse


def main():

    parser = argparse.ArgumentParser(description="Lyrics Generation CLI")

    subparsers = parser.add_subparsers(dest="command")

    # train command
    subparsers.add_parser("train", help="Train LSTM model")

    # baseline command
    subparsers.add_parser("baseline", help="Train baseline n-gram model")

    # full pipeline
    subparsers.add_parser("run", help="Run full pipeline")

    # evaluation command
    subparsers.add_parser("evaluate", help="Train and run held-out/new-data evaluation")

    # generate command
    generate_parser = subparsers.add_parser("generate", help="Generate lyrics")

    generate_parser.add_argument("seed", type=str, help="Seed text for generation")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "run":
        from pipeline.pipeline import LyricsGenerationPipeline
        pipeline = LyricsGenerationPipeline()

        pipeline.run()

    elif args.command == "train":
        from pipeline.pipeline import LyricsGenerationPipeline
        pipeline = LyricsGenerationPipeline()

        pipeline.load_data()
        pipeline.clean_data()
        pipeline.split_data()
        pipeline.tokenize()
        pipeline.build_dataset()
        pipeline.train_lstm()

    elif args.command == "baseline":
        from pipeline.pipeline import LyricsGenerationPipeline
        pipeline = LyricsGenerationPipeline()

        pipeline.load_data()
        pipeline.clean_data()
        pipeline.split_data()
        pipeline.tokenize()
        pipeline.train_baseline()

    elif args.command == "evaluate":
        from pipeline.pipeline import LyricsGenerationPipeline
        pipeline = LyricsGenerationPipeline()

        pipeline.evaluate()

    elif args.command == "generate":
        from pipeline.pipeline import LyricsGenerationPipeline
        pipeline = LyricsGenerationPipeline()
        pipeline.generate(args.seed)


if __name__ == "__main__":
    main()
