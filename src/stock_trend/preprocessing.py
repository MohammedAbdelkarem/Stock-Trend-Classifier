from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
SCALE_COLUMNS = PRICE_COLUMNS + ["Dividends", "Stock Splits"]
FEATURE_COLUMNS = SCALE_COLUMNS + ["Quarter", "DayOfWeek", "Month"]


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["Month"] = result["Date"].dt.month.astype("int8")
    result["DayOfWeek"] = result["Date"].dt.dayofweek.astype("int8")
    result["Quarter"] = result["Date"].dt.quarter.astype("int8")
    return result


def remove_outliers(frame: pd.DataFrame, columns: Iterable[str] = PRICE_COLUMNS) -> pd.DataFrame:
    result = frame.dropna(subset=PRICE_COLUMNS).copy()
    for column in columns:
        first_quartile = result[column].quantile(0.25)
        third_quartile = result[column].quantile(0.75)
        iqr = third_quartile - first_quartile
        lower = first_quartile - 1.5 * iqr
        upper = third_quartile + 1.5 * iqr
        result = result[result[column].between(lower, upper)]
    return result


def prepare_training_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    cleaned = remove_outliers(frame)
    prepared = add_features(cleaned.sort_values(["Ticker", "Date"]))
    scaler = MinMaxScaler(feature_range=(0, 1))
    prepared.loc[:, SCALE_COLUMNS] = scaler.fit_transform(prepared[SCALE_COLUMNS])
    return prepared.reset_index(drop=True), scaler


def prepare_test_frame(
    test_frame: pd.DataFrame,
    raw_training_frame: pd.DataFrame,
    scaler: MinMaxScaler,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = test_frame.sort_values(["ID", "Date"]).copy()
    last_rows = raw_training_frame.sort_values("Date").groupby("Ticker").tail(1)
    merged = test.merge(last_rows, left_on="ID", right_on="Ticker", how="left")
    merged = merged.rename(columns={"Date_x": "Date"})
    merged = add_features(merged)
    features = merged[FEATURE_COLUMNS].copy()
    features.loc[:, SCALE_COLUMNS] = scaler.transform(features[SCALE_COLUMNS])
    return test, features


def make_sequences_full(features: pd.DataFrame, sequence_length: int, pad_mode: str = "edge") -> np.ndarray:
    values = features.to_numpy() if hasattr(features, "to_numpy") else np.asarray(features)
    if values.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if pad_mode == "edge":
        padding = np.repeat(values[:1], sequence_length - 1, axis=0)
    elif pad_mode == "zero":
        padding = np.zeros((sequence_length - 1, values.shape[1]), dtype=values.dtype)
    else:
        raise ValueError("pad_mode must be 'edge' or 'zero'")
    padded = np.vstack([padding, values])
    return np.stack([padded[index:index + sequence_length] for index in range(len(values))])
