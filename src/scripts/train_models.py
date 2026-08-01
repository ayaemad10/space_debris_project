"""
train_models.py
Phase 5 — Model Training & Comparison (CNN vs RNN vs LSTM)

Trains three classifiers on `risk_label` (binary: Pc > 1e-6) using the
engineered feature table, compares Accuracy / F1 / ROC-AUC, and saves the
best model to models/lstm_model.keras (or cnn/rnn, whichever wins).

NOTE: This script requires tensorflow. It was written and reviewed but
NOT executed in the authoring sandbox (no TensorFlow / no network access
there) — run it locally to actually produce models/model_comparison.csv.
A real, executed scikit-learn baseline covering the same split is in
train_baseline.py, which IS verified end-to-end.

Run from project root:
    python src/scripts/train_models.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "engineered_data.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
RANDOM_STATE = 42


def load_split():
    df = pd.read_csv(DATA_PATH)
    y = df["risk_label"].values
    X = df.drop(columns=["risk", "risk_label"]).values.astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    # Sequence models expect (batch, timesteps, features) — one timestep per
    # sample here, since each row is a single conjunction event snapshot.
    X_train_seq = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_seq = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    return X_train_seq, X_test_seq, y_train, y_test, X_train.shape[1]


def build_cnn(n_features: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(1, n_features)),
        layers.Conv1D(64, kernel_size=1, activation="relu"),
        layers.Conv1D(32, kernel_size=1, activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name="cnn_model")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_rnn(n_features: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(1, n_features)),
        layers.SimpleRNN(64, return_sequences=False),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name="rnn_model")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_lstm(n_features: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(1, n_features)),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name="lstm_model")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def evaluate(model, X_test, y_test) -> dict:
    probs = model.predict(X_test, verbose=0).ravel()
    preds = (probs > 0.5).astype(int)
    return {
        "Accuracy": accuracy_score(y_test, preds),
        "F1-Score": f1_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, probs),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, n_features = load_split()
    early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    builders = {"CNN": build_cnn, "RNN": build_rnn, "LSTM": build_lstm}
    results = {}
    trained = {}

    for name, builder in builders.items():
        print(f"\nTraining {name}...")
        model = builder(n_features)
        model.fit(
            X_train, y_train,
            validation_split=0.15,
            epochs=50,
            batch_size=64,
            callbacks=[early_stop],
            verbose=0,
        )
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics
        trained[name] = model
        print(f"{name}: {metrics}")

    comparison = pd.DataFrame(results).T
    comparison.to_csv(OUTPUTS_DIR / "model_comparison.csv")
    print("\nModel comparison:\n", comparison)

    best_name = comparison["ROC-AUC"].idxmax()
    best_model = trained[best_name]
    save_path = MODELS_DIR / f"{best_name.lower()}_model.keras"
    best_model.save(save_path)
    print(f"\nBest model: {best_name} -> saved to {save_path}")


if __name__ == "__main__":
    main()
