"""
shap_explain.py
Phase 6 — SHAP Explainability

NOTE: the `shap` package is not installed in the authoring sandbox and
there is no network access there to install it, so this script was
written and reviewed but NOT executed here. A real, executed stand-in
using the trained model's native feature importances is in
feature_importance_fallback.py (see outputs/shap/feature_importance.png).

Run locally after `pip install shap`:
    python src/scripts/shap_explain.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "engineered_data.csv"
MODEL_PATH = BASE_DIR / "models" / "lstm_model.keras"
OUT_DIR = BASE_DIR / "outputs" / "shap"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    feature_names = [c for c in df.columns if c not in ("risk", "risk_label")]
    X = df[feature_names].values.astype("float32")

    model = tf.keras.models.load_model(MODEL_PATH)

    # LSTM expects (batch, timesteps, features); SHAP needs a flat predict fn.
    background = X[np.random.choice(len(X), 100, replace=False)]

    def predict_fn(flat_x: np.ndarray) -> np.ndarray:
        seq = flat_x.reshape((flat_x.shape[0], 1, flat_x.shape[1]))
        return model.predict(seq, verbose=0).ravel()

    explainer = shap.KernelExplainer(predict_fn, background)
    sample = X[np.random.choice(len(X), 200, replace=False)]
    shap_values = explainer.shap_values(sample, nsamples=100)

    shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary.png", dpi=120)
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
    importance.to_csv(OUT_DIR / "shap_feature_importance.csv")
    print(importance.head(15))


if __name__ == "__main__":
    main()
