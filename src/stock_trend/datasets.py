from collections.abc import Iterator

import numpy as np
import pandas as pd
import tensorflow as tf


def split_by_ticker(
    frame: pd.DataFrame,
    feature_columns: list[str],
    validation_ratio: float,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    train_groups = []
    validation_groups = []
    for _, group in frame.groupby("Ticker", sort=False):
        group = group.reset_index(drop=True)
        split = int(len(group) * (1 - validation_ratio))
        train_part, validation_part = group.iloc[:split], group.iloc[split:]
        train_groups.append(
            (train_part[feature_columns].to_numpy(dtype=np.float32), train_part["Close"].to_numpy(dtype=np.float32))
        )
        validation_groups.append(
            (validation_part[feature_columns].to_numpy(dtype=np.float32), validation_part["Close"].to_numpy(dtype=np.float32))
        )
    return train_groups, validation_groups


def sequence_generator(
    groups: list[tuple[np.ndarray, np.ndarray]],
    lookback: int,
    horizon: int,
    batch_size: int,
    stride: int = 1,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    features_batch, labels_batch = [], []
    for features, close in groups:
        max_start = len(close) - (lookback + horizon) + 1
        for start in range(0, max(0, max_start), stride):
            end = start + lookback - 1
            features_batch.append(features[start:start + lookback])
            labels_batch.append(float(close[end + horizon] > close[end]))
            if len(features_batch) == batch_size:
                yield np.stack(features_batch), np.asarray(labels_batch, dtype=np.float32)
                features_batch, labels_batch = [], []
    if features_batch:
        yield np.stack(features_batch), np.asarray(labels_batch, dtype=np.float32)


def create_dataset(groups, lookback: int, horizon: int, feature_count: int, batch_size: int) -> tf.data.Dataset:
    signature = (
        tf.TensorSpec(shape=(None, lookback, feature_count), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: sequence_generator(groups, lookback, horizon, batch_size),
        output_signature=signature,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def collect_labels(dataset: tf.data.Dataset, steps: int) -> np.ndarray:
    labels = []
    for _, batch_labels in zip(range(steps), dataset):
        labels.append(batch_labels.numpy())
    if not labels:
        raise ValueError("No sequences were generated. Add more rows per ticker or reduce lookback/horizon.")
    return np.concatenate(labels)
