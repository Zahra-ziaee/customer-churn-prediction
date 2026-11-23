
from pathlib import Path

import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE
from src.data_loader import load_raw_data
from src.preprocessing import (
    prepare_churn_dataset,
    split_features_target,
    train_test_split_data,
)
from src.models import get_models


OUTPUT_FILE = Path(
    "results/feature_importance_25features_permutation.csv"
)


def main():
    # Load and prepare dataset.
    raw_df, _, _ = load_raw_data()

    processed_df = prepare_churn_dataset(raw_df)

    X, y = split_features_target(processed_df)

    # Keep the original test partition out of this analysis.
    X_train, _, y_train, _ = train_test_split_data(X, y)

    # Create an internal validation split from training data.
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    # Train Gradient Boosting on the inner training split.
    model = get_models(X_fit)["Gradient Boosting"]
    model.fit(X_fit, y_fit)

    print("Calculating permutation importance...")
    print(f"Number of features: {X.shape[1]}")
    print(f"Training samples: {len(X_fit)}")
    print(f"Validation samples: {len(X_valid)}")

    # Measure importance using validation ROC-AUC.
    importance = permutation_importance(
        model,
        X_valid,
        y_valid,
        scoring="roc_auc",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    results = pd.DataFrame({
        "feature": X_valid.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    })

    results = results.sort_values(
        by="importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(OUTPUT_FILE, index=False)

    print("\nFeature Importance Results:")
    print(results.round(5).to_string(index=False))

    print(f"\nSaved to: {OUTPUT_FILE}")

    print(
        "\nNote: Importance was calculated on an "
        "internal training-only validation split. "
        "The original test partition was not used."
    )


if __name__ == "__main__":
    main()
