import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix


def confusion_matrix_from_dataset(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    steps: int,
    threshold: float = 0.5,
) -> np.ndarray:
    actual, predicted = [], []
    for _, (features, labels) in zip(range(steps), dataset):
        probabilities = model.predict(features, verbose=0).reshape(-1)
        actual.append(labels.numpy())
        predicted.append(probabilities)
    if not actual:
        raise ValueError("The evaluation dataset contains no batches.")
    y_true = np.concatenate(actual).astype(int)
    y_pred = (np.concatenate(predicted) >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
