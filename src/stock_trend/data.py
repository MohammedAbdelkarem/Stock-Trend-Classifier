from pathlib import Path

import pandas as pd


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing input file: {path}. Put the Kaggle CSV files in the data directory."
        )


def load_training_data(path: Path) -> pd.DataFrame:
    _require_file(path)
    return pd.read_csv(path, parse_dates=["Date"])


def load_test_data(path: Path) -> pd.DataFrame:
    _require_file(path)
    return pd.read_csv(path, parse_dates=["Date"])


def load_sample_submission(path: Path) -> pd.DataFrame:
    _require_file(path)
    return pd.read_csv(path)


def save_submission(submission: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
