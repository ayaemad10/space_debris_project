"""
eda.py
Phase 3 — Exploratory Data Analysis

Generates and saves:
    outputs/eda/risk_distribution.png
    outputs/eda/miss_distance_vs_risk.png
    outputs/eda/correlation_heatmap.png
    outputs/eda/class_balance.png

Run from project root:
    python src/scripts/eda.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "cleaned_data.csv"
OUT_DIR = BASE_DIR / "outputs" / "eda"


def run_eda(data_path: Path = DATA_PATH, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)

    # 1. Risk distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["risk"], bins=50, color="#3b6ea5", edgecolor="white")
    ax.axvline(-6, color="crimson", linestyle="--", label="high-risk cut (Pc=1e-6)")
    ax.set_xlabel("risk (log10 Pc)")
    ax.set_ylabel("count")
    ax.set_title("Collision Risk Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "risk_distribution.png", dpi=120)
    plt.close(fig)

    # 2. Miss distance vs risk
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["miss_distance"], df["risk"], s=6, alpha=0.4, c="#e07b39")
    ax.set_xlabel("miss_distance (scaled)")
    ax.set_ylabel("risk (log10 Pc)")
    ax.set_title("Miss Distance vs Collision Risk")
    fig.tight_layout()
    fig.savefig(out_dir / "miss_distance_vs_risk.png", dpi=120)
    plt.close(fig)

    # 3. Correlation heatmap on a meaningful subset (full 103x103 is unreadable)
    subset_cols = [
        "miss_distance", "relative_speed", "relative_pos_mag", "relative_vel_mag",
        "max_risk_estimate", "max_risk_scaling", "mahalanobis_distance",
        "t_j2k_ecc", "c_j2k_ecc", "t_j2k_inc", "c_j2k_inc", "risk",
    ]
    subset_cols = [c for c in subset_cols if c in df.columns]
    corr = df[subset_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(subset_cols)))
    ax.set_xticklabels(subset_cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(subset_cols)))
    ax.set_yticklabels(subset_cols, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Heatmap (key orbital/conjunction features)")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=120)
    plt.close(fig)

    # 4. Class balance
    counts = df["risk_label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(["LOW/normal (0)", "HIGH risk (1)"], counts.values, color=["#3b6ea5", "#c0392b"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, str(v), ha="center")
    ax.set_title("Class Balance (risk_label)")
    fig.tight_layout()
    fig.savefig(out_dir / "class_balance.png", dpi=120)
    plt.close(fig)

    print(f"Saved 4 EDA figures to {out_dir}")


if __name__ == "__main__":
    run_eda()
