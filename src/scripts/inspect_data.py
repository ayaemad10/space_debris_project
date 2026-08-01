"""
inspect_data.py
Phase 1 — Data Understanding

Loads the raw engineered dataset and prints a schema/quality report.
No cleaning or transformation happens here — this is read-only inspection.

Run from project root:
    python src/scripts/inspect_data.py
"""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "engineered_data_sample.csv"


def inspect(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    print(f"Loaded: {path.name}")
    print(f"Shape:  {df.shape[0]} rows x {df.shape[1]} columns")
    print()

    n_missing = int(df.isnull().sum().sum())
    n_dupes = int(df.duplicated().sum())
    print(f"Missing values total: {n_missing}")
    print(f"Duplicate rows:       {n_dupes}")
    print()

    print("Target column: 'risk' (log10 collision probability, ESA Kelvins convention)")
    print(df["risk"].describe())
    print()
    print(f"Rows floored at -30 (no meaningful conjunction): {(df['risk'] == -30).sum()}")
    print(f"Rows with risk > -6 (Pc > 1e-6, conventional 'high risk' cut): "
          f"{(df['risk'] > -6).sum()} ({(df['risk'] > -6).mean() * 100:.2f}%)")

    return df


if __name__ == "__main__":
    inspect()
