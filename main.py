import argparse
from pathlib import Path

from stock_trend.config import ProjectConfig
from stock_trend.pipeline import train_and_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the stock trend classifier and create a submission.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing train.csv, test.csv, and sample_submission.csv.")
    parser.add_argument("--epochs", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = ProjectConfig()
    config = ProjectConfig(data_dir=args.data_dir or defaults.data_dir, epochs=args.epochs)
    output = train_and_predict(config)
    print(f"Submission written to {output}")


if __name__ == "__main__":
    main()
