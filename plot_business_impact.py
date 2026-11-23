
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "business_impact_scenarios.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "business_impact_by_threshold.png"
)


# =====================================================
# BUSINESS IMPACT VISUALIZATION
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
        "scenario",
        "threshold",
        "estimated_net_value",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError(
            "Business impact data is empty."
        )

    # -------------------------------------------------
    # CREATE CHART
    # -------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    scenario_colors = {
        "High outreach cost": "#2878B5",
        "Low outreach cost": "#ED7D20",
    }

    # -------------------------------------------------
    # PLOT BUSINESS SCENARIOS
    # -------------------------------------------------

    for scenario, group in df.groupby("scenario"):

        group = group.sort_values(
            by="threshold"
        )

        color = scenario_colors.get(
            scenario,
            "#555555",
        )

        ax.plot(
            group["threshold"] * 100,
            group["estimated_net_value"],
            marker="o",
            linewidth=2.5,
            markersize=7,
            label=scenario,
            color=color,
        )

        # Identify the highest estimated value.
        best = group.loc[
            group["estimated_net_value"].idxmax()
        ]

        best_x = float(
            best["threshold"] * 100
        )

        best_y = float(
            best["estimated_net_value"]
        )

        # ---------------------------------------------
        # POSITION ANNOTATIONS WITHOUT OVERLAP
        # ---------------------------------------------

        if scenario == "Low outreach cost":

            # Position below the orange peak.
            offset = (12, -30)
            horizontal_align = "left"
            vertical_align = "top"

        else:

            # Position above the blue peak.
            offset = (0, 18)
            horizontal_align = "center"
            vertical_align = "bottom"

        ax.annotate(
            f"Best: {best['threshold']:.0%}\n"
            f"Value: {best_y:,.0f}",
            xy=(best_x, best_y),
            xytext=offset,
            textcoords="offset points",
            ha=horizontal_align,
            va=vertical_align,
            fontsize=10,
            fontweight="bold",
            color="#263746",
        )

    # -------------------------------------------------
    # TITLES AND AXES
    # -------------------------------------------------

    ax.set_title(
        "Estimated Business Impact by Screening Threshold",
        fontsize=16,
        fontweight="bold",
        pad=24,
    )

    ax.set_xlabel(
        "Screening Threshold (%)",
        fontsize=12,
        labelpad=12,
    )

    ax.set_ylabel(
        "Estimated Net Value (Hypothetical Units)",
        fontsize=12,
        labelpad=12,
    )

    ax.tick_params(
        axis="both",
        labelsize=10,
    )

    # -------------------------------------------------
    # GRID AND LEGEND
    # -------------------------------------------------

    ax.grid(
        alpha=0.2,
        linestyle="--",
    )

    ax.set_axisbelow(True)

    ax.legend(
        title="Business Scenario",
        fontsize=10,
        title_fontsize=11,
        loc="upper right",
        frameon=True,
    )

    # -------------------------------------------------
    # AXIS LIMITS
    # -------------------------------------------------

    ax.set_xlim(
        left=df["threshold"].min() * 100 - 3,
        right=df["threshold"].max() * 100 + 3,
    )

    value_min = df["estimated_net_value"].min()
    value_max = df["estimated_net_value"].max()

    value_range = value_max - value_min

    ax.set_ylim(
        bottom=value_min - value_range * 0.08,
        top=value_max + value_range * 0.12,
    )

    # -------------------------------------------------
    # EXPLANATORY FOOTNOTE
    # -------------------------------------------------

    fig.text(
        0.5,
        0.025,
        "Illustrative assumptions only. "
        "Not measured campaign profit or causal impact.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#555555",
    )

    # -------------------------------------------------
    # SAVE FIGURE
    # -------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout(
        rect=[0, 0.07, 1, 1]
    )

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Business impact chart created successfully."
    )

    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
