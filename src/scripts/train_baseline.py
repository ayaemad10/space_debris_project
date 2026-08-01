"""
train_baseline.py
Pipeline validation baseline (NOT the deliverable CNN/RNN/LSTM comparison).

TensorFlow is unavailable in the environment this project was authored in,
so the actual Keras training in train_models.py could not be executed here.
This script trains a scikit-learn GradientBoostingClassifier on the exact
same train/test split logic to prove the data pipeline (clean -> feature
engineer -> split -> train -> evaluate) runs correctly end-to-end, with
real, reproducible numbers.

Run from project root:
    python src/scripts/train_baseline.py
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "engineered_data.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
RANDOM_STATE = 42


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    y = df["risk_label"].values
    X = df.drop(columns=["risk", "risk_label"]).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    metrics = {
        "Accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "F1-Score": round(float(f1_score(y_test, preds)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_test, probs)), 4),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "positive_rate_test": round(float(y_test.mean()), 4),
    }
    print(json.dumps(metrics, indent=2))

    with open(OUTPUTS_DIR / "baseline_validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    import joblib
    joblib.dump(clf, BASE_DIR / "models" / "baseline_gbc.joblib")
    print(f"\nSaved baseline model to models/baseline_gbc.joblib")
    print(f"Saved metrics to outputs/baseline_validation_metrics.json")


if __name__ == "__main__":
    main()
