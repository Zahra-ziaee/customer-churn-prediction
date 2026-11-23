
from pathlib import Path

import pandas as pd

from src.data_loader import load_raw_data


OUTPUT_FILE = Path(
    "results/churn_segmentation_by_tenure_contract.csv"
)


def main():
    # Load original data for business analysis.
    raw_df, _, _ = load_raw_data()
    df = raw_df.copy()

    # Create binary churn indicator.
    df["Churn"] = (
        df["Customer Status"] == "Churned"
    ).astype(int)

    # Group customers by tenure.
    df["Tenure Group"] = pd.cut(
        df["Tenure in Months"],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=["0-12", "13-24", "25-48", "49+"],
    )

    # Validate segmentation coverage.
    if df["Tenure Group"].isna().any():
        raise ValueError(
            "Some customers have missing or invalid tenure."
        )

    if df["Contract"].isna().any():
        raise ValueError(
            "Some customers have missing contract values."
        )

    # Calculate segment-level churn statistics.
    segments = (
        df.groupby(
            ["Tenure Group", "Contract"],
            observed=True,
        )["Churn"]
        .agg(
            customers="size",
            churned="sum",
            churn_rate="mean",
        )
        .reset_index()
    )

    total_churned = int(df["Churn"].sum())

    segments["share_of_all_churn"] = (
        segments["churned"] / total_churned
    )

    # Rank segments by total churn cases.
    segments = segments.sort_values(
        by="churned",
        ascending=False,
    )

    # Save results.
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    segments.to_csv(OUTPUT_FILE, index=False)

    print("\nCustomer Segmentation Analysis")
    print("=" * 60)

    print(f"Total customers: {len(df):,}")
    print(f"Total churned: {total_churned:,}")

    print("\nSegments ranked by churn count:")
    print(segments.round(4).to_string(index=False))

    print("\nHighest churn-count segment:")
    print(segments.iloc[0].to_string())

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
