"""
preprocess.py
Phase 2 — Data Cleaning & Preprocessing

`engineered_data_sample.csv` arrives already numeric, already scaled
(all 103 columns are float64, z-score-like) and has zero missing values
and zero duplicate rows — it is the *output* of an upstream feature-
engineering step, not raw CDM data.

So "cleaning" here is verification, not imputation:
    1. Confirm no missing values / duplicates (fail loudly if that changes).
    2. Derive the binary classification target `risk_label` from the
       continuous `risk` column (log10 collision probability), since
       CNN/RNN/LSTM are being trained as classifiers per the project spec.
       Threshold: risk > -6  <=>  Pc > 1e-6, the conventional ESA Kelvins
       "high risk" cut used in the original challenge scoring.
    3. Save the result as cleaned_data.csv.

Run from project root:
    python src/scripts/preprocess.py
"""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_PATH = BASE_DIR / "data" / "engineered_data_sample.csv"
OUT_PATH = BASE_DIR / "data" / "cleaned_data.csv"

RISK_THRESHOLD = -6.0  # log10(Pc); Pc > 1e-6


def preprocess(raw_path: Path = RAW_PATH, out_path: Path = OUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    n_missing = int(df.isnull().sum().sum())
    n_dupes = int(df.duplicated().sum())
    if n_missing:
        # Data is expected to be pre-cleaned; if this ever triggers on a
        # different data drop, impute numerics with the median rather than
        # silently dropping rows.
        df = df.fillna(df.median(numeric_only=True))
        print(f"WARNING: found {n_missing} missing values — imputed with column median.")
    if n_dupes:
        df = df.drop_duplicates()
        print(f"WARNING: dropped {n_dupes} duplicate rows.")

    df["risk_label"] = (df["risk"] > RISK_THRESHOLD).astype(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(df)} | Positive class (risk_label=1): {df['risk_label'].sum()} "
          f"({df['risk_label'].mean() * 100:.2f}%)")
    return df


if __name__ == "__main__":
    preprocess()
