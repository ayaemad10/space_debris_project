"""
decision_engine.py
Phase 7 — Decision Engine

Maps a classifier's predicted probability (0.0-1.0, the model's confidence
that risk_label=1, i.e. Pc > 1e-6) to a risk level and recommended action.

Thresholds operate on the *model's output probability*, not on the raw
`risk` (log10 Pc) column — those live on totally different scales, which
is a bug in the originally uploaded app.py/mission_control.py (they treat
`risk` itself as if it were already a 0-1 probability).

    0.0 - 0.5  -> LOW       -> NO ACTION REQUIRED
    0.5 - 0.8  -> HIGH      -> INCREASE MONITORING
    0.8 - 1.0  -> CRITICAL  -> IMMEDIATE MANEUVER

Run from project root:
    python src/scripts/decision_engine.py
"""
from dataclasses import dataclass


@dataclass
class Decision:
    probability: float
    risk_level: str
    action: str


def decide(probability: float) -> Decision:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")

    if probability > 0.8:
        return Decision(probability, "CRITICAL", "IMMEDIATE MANEUVER REQUIRED")
    elif probability > 0.5:
        return Decision(probability, "HIGH", "INCREASE MONITORING FREQUENCY")
    else:
        return Decision(probability, "LOW", "NO ACTION REQUIRED")


if __name__ == "__main__":
    from pathlib import Path
    import joblib
    import pandas as pd

    BASE_DIR = Path(__file__).resolve().parents[2]
    df = pd.read_csv(BASE_DIR / "data" / "engineered_data.csv")
    clf = joblib.load(BASE_DIR / "models" / "baseline_gbc.joblib")

    feature_cols = [c for c in df.columns if c not in ("risk", "risk_label")]
    sample = df[feature_cols].iloc[:5]
    probs = clf.predict_proba(sample)[:, 1]

    for i, p in enumerate(probs):
        d = decide(float(p))
        print(f"Event {i}: p={d.probability:.3f} -> {d.risk_level} -> {d.action}")
