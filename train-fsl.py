"""
Train a Conv1D + LSTM model on the landmark-based FSL alphabet dataset
(X.npy / y.npy produced by convert_fsl_to_npy.py).

Input shape per sample: (21, 3)  -> 21 hand landmarks, each with x, y, z

Outputs after running:
    fsl_model.keras       -> trained model, used by fsl_camera_recognition.py
    label_classes.npy     -> class label order, used by fsl_camera_recognition.py

Requirements:
    pip install tensorflow scikit-learn numpy matplotlib
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
X_PATH = "X.npy"
Y_PATH = "y.npy"
MODEL_OUT_PATH = "fsl_model.keras"
LABELS_OUT_PATH = "label_classes.npy"

TEST_SIZE = 0.2
VAL_SIZE = 0.1          # taken from the training split
RANDOM_STATE = 42
EPOCHS = 60
BATCH_SIZE = 32
# -----------------------------------------


def build_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape),           # (21, 3)

        layers.Conv1D(32, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    # ---- Load data ----
    X = np.load(X_PATH)      # shape (N, 21, 3)
    y_raw = np.load(Y_PATH)  # shape (N,) string labels

    print(f"Loaded X: {X.shape}, y: {y_raw.shape}")

    # ---- Encode labels ----
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)   # letters -> integers
    np.save(LABELS_OUT_PATH, encoder.classes_)
    print(f"Classes ({len(encoder.classes_)}): {list(encoder.classes_)}")

    # ---- Normalize landmarks ----
    # Landmarks from MediaPipe are already roughly in 0-1 range for x/y,
    # but z can vary. Standardize all features for stable training.
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X - mean) / std
    np.save("norm_mean.npy", mean)
    np.save("norm_std.npy", std)

    # ---- Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train
    )

    print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    # ---- Build model ----
    model = build_model(input_shape=X.shape[1:], num_classes=len(encoder.classes_))
    model.summary()

    # ---- Callbacks ----
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    # ---- Train ----
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    # ---- Evaluate ----
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc*100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")

    # ---- Save model ----
    model.save(MODEL_OUT_PATH)
    print(f"Model saved to {MODEL_OUT_PATH}")
    print(f"Label classes saved to {LABELS_OUT_PATH}")
    print("Normalization stats saved to norm_mean.npy / norm_std.npy")

    # ---- Plot training curves ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    print("Training curves saved to training_history.png")


if __name__ == "__main__":
    main()