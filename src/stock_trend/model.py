import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Conv1D, Dense, Dropout, GlobalAveragePooling1D, LSTM, MaxPooling1D, SpatialDropout1D
from tensorflow.keras.metrics import AUC, Precision, Recall


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    def loss(y_true, y_pred):
        clipped = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        return -tf.reduce_mean(
            alpha * y_true * tf.pow(1 - clipped, gamma) * tf.math.log(clipped)
            + (1 - alpha) * (1 - y_true) * tf.pow(clipped, gamma) * tf.math.log(1 - clipped)
        )
    return loss


def build_model(sequence_length: int, feature_count: int) -> tf.keras.Model:
    model = Sequential([
        Conv1D(64, 3, padding="same", activation="relu", input_shape=(sequence_length, feature_count)),
        BatchNormalization(),
        Conv1D(64, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        SpatialDropout1D(0.2),
        Conv1D(128, 3, padding="same", activation="relu"),
        BatchNormalization(),
        Conv1D(128, 3, padding="same", activation="relu"),
        BatchNormalization(),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
        GlobalAveragePooling1D(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=focal_loss(),
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall"), AUC(name="auc_roc"), AUC(name="auc_pr", curve="PR")],
    )
    return model
