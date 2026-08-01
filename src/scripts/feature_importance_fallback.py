"""
feature_importance_fallback.py
Real, executed explainability pass using the trained baseline model's
native feature importances (GradientBoostingClassifier.feature_importances_),
since `shap` could not be installed in this sandbox (no network access).
This is a legitimate stand-in, not a substitute for shap.summary_plot on
the actual deliverable model — run shap_explain.py locally for that.

Run from project root:
    python src/scripts/feature_importance_fallback.py
"""
from pathlib import Path
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "engineered_data.csv"
MODEL_PATH = BASE_DIR / "models" / "baseline_gbc.joblib"
OUT_DIR = BASE_DIR / "outputs" / "shap"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    feature_names = [c for c in df.columns if c not in ("risk", "risk_label")]

    clf = joblib.load(MODEL_PATH)
    importance = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
    importance.to_csv(OUT_DIR / "feature_importance.csv")

    top15 = importance.head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top15.index[::-1], top15.values[::-1], color="#3b6ea5")
    ax.set_xlabel("Feature importance (Gini-based)")
    ax.set_title("Top 15 Most Important Features — Collision Risk Classifier")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)

    print(top15)


if __name__ == "__main__":
    main()
