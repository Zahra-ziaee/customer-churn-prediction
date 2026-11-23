
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data_loader import load_raw_data
from src.preprocessing import (
    prepare_churn_dataset,
    split_features_target,
    train_test_split_data,
)
from src.models import get_models
from src.config import RANDOM_STATE


def main():
    raw_df, _, _ = load_raw_data()

    processed_df = prepare_churn_dataset(raw_df)
    X, y = split_features_target(processed_df)

    X_train, _, y_train, _ = train_test_split_data(X, y)

    print(f"Model features: {X.shape[1]}")
    print(f"Training samples: {len(X_train)}")
    print("Test data will not be used for model selection.")

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    results = []

    for name, pipeline in get_models(X_train).items():
        print(f"\nEvaluating: {name}")

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        row = {"model": name}

        for metric in scoring:
            row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
            row[f"{metric}_std"] = scores[f"test_{metric}"].std()

        results.append(row)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="roc_auc_mean",
        ascending=False,
    )

    output = Path("results/model_cv_25features.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output, index=False)

    print("\nCross-validation results:")
    print(results_df.round(4).to_string(index=False))
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
