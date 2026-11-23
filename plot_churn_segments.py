
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_FILE = Path(
    "results/churn_segmentation_by_tenure_contract.csv"
)

OUTPUT_FILE = Path(
    "results/figures/churn_rate_by_contract_tenure.png"
)


def main():
    df = pd.read_csv(INPUT_FILE)

    tenure_order = ["0-12", "13-24", "25-48", "49+"]
    contract_order = [
        "Month-to-Month",
        "One Year",
        "Two Year",
    ]

    rates = df.pivot(
        index="Contract",
        columns="Tenure Group",
        values="churn_rate",
    ).reindex(
        index=contract_order,
        columns=tenure_order,
    )

    customers = df.pivot(
        index="Contract",
        columns="Tenure Group",
        values="customers",
    ).reindex(
        index=contract_order,
        columns=tenure_order,
    )

    churned = df.pivot(
        index="Contract",
        columns="Tenure Group",
        values="churned",
    ).reindex(
        index=contract_order,
        columns=tenure_order,
    )

    if rates.isna().any().any():
        raise ValueError("Missing contract-tenure segments.")

    fig, ax = plt.subplots(figsize=(12, 5.5))

    heatmap = ax.imshow(
        rates.to_numpy(),
        cmap="YlOrRd",
        vmin=0,
        vmax=0.6,
        aspect="auto",
    )

    ax.set_xticks(range(len(tenure_order)))
    ax.set_xticklabels(tenure_order)

    ax.set_yticks(range(len(contract_order)))
    ax.set_yticklabels(contract_order)

    ax.set_xlabel("Customer Tenure (Months)")
    ax.set_ylabel("Contract Type")

    ax.set_title(
        "Customer Churn Rate by Contract and Tenure",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    for i in range(len(contract_order)):
        for j in range(len(tenure_order)):
            rate = rates.iloc[i, j]
            count = int(customers.iloc[i, j])
            churn_count = int(churned.iloc[i, j])

            label = (
                f"{rate:.1%}\n"
                f"{churn_count:,} / {count:,} churned"
            )

            text_color = (
                "white" if rate > 0.35 else "black"
            )

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

    colorbar = fig.colorbar(
        heatmap,
        ax=ax,
        fraction=0.035,
        pad=0.04,
    )

    colorbar.set_label("Churn Rate")

    fig.tight_layout()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Segmentation heatmap created successfully.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
