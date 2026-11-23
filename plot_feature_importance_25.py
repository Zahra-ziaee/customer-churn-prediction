
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# =====================================================
# FILE PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "feature_importance_25features_permutation.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "feature_importance_25features_permutation.png"
)


# =====================================================
# FEATURE IMPORTANCE VISUALIZATION
# =====================================================

def main():

    # -------------------------------------------------
    # LOAD AND VALIDATE DATA
    # -------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "feature",
        "importance_mean",
        "importance_std",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError(
            "Feature importance data is empty."
        )

    # -------------------------------------------------
    # SELECT TOP 12 FEATURES
    # -------------------------------------------------

    top_features = (
        df.sort_values(
            by="importance_mean",
            ascending=False,
        )
        .head(12)
        .sort_values(
            by="importance_mean",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------
    # CREATE CHART
    # -------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 7.5))

    ax.barh(
        top_features["feature"],
        top_features["importance_mean"],
        xerr=top_features["importance_std"],
        capsize=4,
        color="#467FA8",
        alpha=0.9,
        ecolor="#263746",
        error_kw={
            "elinewidth": 1.3,
            "capthick": 1.3,
        },
    )

    ax.axvline(
        x=0,
        color="black",
        linewidth=0.8,
    )

    # -------------------------------------------------
    # TITLES AND LABELS
    # -------------------------------------------------

    ax.set_title(
        "Top 12 Features - Permutation Importance",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    ax.set_xlabel(
        "Mean Decrease in Validation ROC-AUC",
        fontsize=11,
        labelpad=12,
    )

    ax.set_ylabel("")

    ax.tick_params(
        axis="both",
        labelsize=10,
    )

    # -------------------------------------------------
    # GRID
    # -------------------------------------------------

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.2,
    )

    ax.set_axisbelow(True)

    # -------------------------------------------------
    # ADD VALUES AFTER ERROR BARS
    # -------------------------------------------------

    for index, row in top_features.iterrows():

        value = float(row["importance_mean"])
        std = float(row["importance_std"])

        ax.annotate(
            f"{value:.4f}",
            xy=(value + std, index),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color="#263746",
        )

    # -------------------------------------------------
    # AXIS LIMITS
    # -------------------------------------------------

    max_importance_with_error = (
        top_features["importance_mean"]
        + top_features["importance_std"]
    ).max()

    ax.set_xlim(
        left=0,
        right=max_importance_with_error * 1.30,
    )

    # -------------------------------------------------
    # EXPLANATORY FOOTNOTE
    # -------------------------------------------------

    fig.text(
        0.5,
        0.015,
        "Permutation importance measured on an internal "
        "validation split using ROC-AUC (5 repeats). "
        "Results indicate predictive relevance, not causation.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )

    # -------------------------------------------------
    # SAVE FIGURE
    # -------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.subplots_adjust(
        left=0.26,
        right=0.96,
        top=0.90,
        bottom=0.14,
    )

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Permutation feature importance chart "
        "created successfully."
    )

    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
