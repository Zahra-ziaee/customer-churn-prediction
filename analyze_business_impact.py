
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("results/threshold_cv_25features.csv")
OUTPUT_FILE = Path("results/business_impact_scenarios.csv")

# Hypothetical assumptions, not observed business outcomes.
CUSTOMER_VALUE = 200.0
RETENTION_SUCCESS_RATE = 0.10

SCENARIOS = {
    "Low outreach cost": 5.0,
    "High outreach cost": 10.0,
}


def main():
    results = pd.read_csv(INPUT_FILE)
    business_rows = []

    for scenario_name, contact_cost in SCENARIOS.items():
        for _, row in results.iterrows():
            expected_saved = (
                row["TP"] * RETENTION_SUCCESS_RATE
            )

            expected_value = (
                expected_saved * CUSTOMER_VALUE
            )

            outreach_cost = (
                row["flagged_customers"] * contact_cost
            )

            estimated_net_value = (
                expected_value - outreach_cost
            )

            business_rows.append({
                "scenario": scenario_name,
                "threshold": row["threshold"],
                "contact_cost": contact_cost,
                "flagged_customers": int(
                    row["flagged_customers"]
                ),
                "true_positives": int(row["TP"]),
                "false_positives": int(row["FP"]),
                "expected_saved_customers": expected_saved,
                "estimated_net_value": estimated_net_value,
            })

    business_df = pd.DataFrame(business_rows)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    business_df.to_csv(OUTPUT_FILE, index=False)

    print("\nHypothetical business impact analysis:")
    print(business_df.round(2).to_string(index=False))

    print("\nBest threshold by scenario:")

    for scenario_name, group in business_df.groupby("scenario"):
        best = group.loc[
            group["estimated_net_value"].idxmax()
        ]

        print(
            f"{scenario_name}: "
            f"threshold={best['threshold']:.2f}, "
            f"estimated_net_value="
            f"{best['estimated_net_value']:.2f}"
        )

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(
        "\nIMPORTANT: These results use illustrative "
        "assumptions, not measured causal retention effects."
    )


if __name__ == "__main__":
    main()
