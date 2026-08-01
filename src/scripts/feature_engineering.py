"""
feature_engineering.py
Phase 4 — Feature Engineering

`engineered_data_sample.csv` already contains the derived features described
in the project README:
    relative_pos_mag = sqrt(r^2 + t^2 + n^2)   (RTN relative position magnitude)
    relative_vel_mag = sqrt(vr^2 + vt^2 + vn^2) (RTN relative velocity magnitude)

This script verifies those formulas actually hold against the RTN components
already in the file (rather than trusting the column names blindly), then
writes the final X/y split used by training.

Run from project root:
    python src/scripts/feature_engineering.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
IN_PATH = BASE_DIR / "data" / "cleaned_data.csv"
OUT_PATH = BASE_DIR / "data" / "engineered_data.csv"

DROP_COLS = ["risk", "risk_label"]  # target-derived columns, never features


def verify_engineered_features(df: pd.DataFrame) -> None:
    pos_check = np.sqrt(
        df["relative_position_r"] ** 2
        + df["relative_position_t"] ** 2
        + df["relative_position_n"] ** 2
    )
    vel_check = np.sqrt(
        df["relative_velocity_r"] ** 2
        + df["relative_velocity_t"] ** 2
        + df["relative_velocity_n"] ** 2
    )
    pos_diff = (df["relative_pos_mag"] - pos_check).abs().max()
    vel_diff = (df["relative_vel_mag"] - vel_check).abs().max()
    print(f"Max |relative_pos_mag - sqrt(r^2+t^2+n^2)| = {pos_diff:.2e}")
    print(f"Max |relative_vel_mag - sqrt(vr^2+vt^2+vn^2)| = {vel_diff:.2e}")
    assert pos_diff < 1e-6, "relative_pos_mag formula does not match RTN components"
    assert vel_diff < 1e-6, "relative_vel_mag formula does not match RTN components"
    print("Feature formulas verified against source columns.")


def build_features(in_path: Path = IN_PATH, out_path: Path = OUT_PATH):
    df = pd.read_csv(in_path)
    verify_engineered_features(df)

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    ordered = feature_cols + DROP_COLS
    df = df[ordered]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({df.shape[0]} rows, {len(feature_cols)} features + target)")
    return df


if __name__ == "__main__":
    build_features()
