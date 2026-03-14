"""
main.py

Command line interface for the lyrics generation project.
"""
import argparse
from pipeline.pipeline import LyricsGenerationPipeline


def main():

    parser = argparse.ArgumentParser(description="Lyrics Generation CLI")

    subparsers = parser.add_subparsers(dest="command")

    # train command
    subparsers.add_parser("train", help="Train LSTM model")

    # baseline command
    subparsers.add_parser("baseline", help="Train baseline n-gram model")

    # full pipeline
    subparsers.add_parser("run", help="Run full pipeline")

    # generate command
    generate_parser = subparsers.add_parser("generate", help="Generate lyrics")

    generate_parser.add_argument("seed", type=str, help="Seed text for generation")

    args = parser.parse_args()

    pipeline = LyricsGenerationPipeline()

    if args.command == "run":

        pipeline.run()

    elif args.command == "train":

        pipeline.load_data()
        pipeline.clean_data()
        pipeline.tokenize()
        pipeline.build_dataset()
        pipeline.train_lstm()

    elif args.command == "baseline":

        pipeline.load_data()
        pipeline.clean_data()
        pipeline.tokenize()
        pipeline.train_baseline()

    elif args.command == "generate":

        pipeline.load_data()
        pipeline.clean_data()
        pipeline.tokenize()
        pipeline.build_dataset()
        pipeline.train_lstm()

        pipeline.generate(args.seed)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
