import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from .config import ProjectConfig
from .data import load_sample_submission, load_test_data, load_training_data, save_submission
from .datasets import collect_labels, create_dataset, split_by_ticker
from .model import build_model
from .preprocessing import FEATURE_COLUMNS, make_sequences_full, prepare_test_frame, prepare_training_frame


def train_and_predict(config: ProjectConfig):
    config.create_directories()
    raw_train = load_training_data(config.train_path)
    train_frame, scaler = prepare_training_frame(raw_train)
    train_groups, validation_groups = split_by_ticker(train_frame, FEATURE_COLUMNS, config.validation_ratio)
    train_dataset = create_dataset(train_groups, config.lookback, config.horizon, len(FEATURE_COLUMNS), config.batch_size)
    validation_dataset = create_dataset(validation_groups, config.lookback, config.horizon, len(FEATURE_COLUMNS), config.batch_size)

    labels = collect_labels(train_dataset, config.steps_per_epoch)
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels.astype(int))
    class_weights = dict(zip(classes, weights))

    model = build_model(config.lookback, len(FEATURE_COLUMNS))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc_pr", mode="max", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(config.model_path, monitor="val_auc_pr", mode="max", save_best_only=True),
    ]
    model.fit(
        train_dataset,
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=config.validation_steps,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=False,
    )

    test = load_test_data(config.test_path)
    _, test_features = prepare_test_frame(test, raw_train, scaler)
    sequences = make_sequences_full(test_features, config.lookback)
    probabilities = model.predict(sequences, verbose=0).reshape(-1)
    submission = load_sample_submission(config.sample_submission_path)
    submission = submission.assign(Pred=(probabilities >= config.threshold).astype(int))
    save_submission(submission[["ID", "Pred"]], config.submission_path)
    return config.submission_path
