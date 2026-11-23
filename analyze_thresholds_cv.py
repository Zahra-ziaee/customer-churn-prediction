
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.config import RANDOM_STATE
from src.data_loader import load_raw_data
from src.preprocessing import (
    prepare_churn_dataset,
    split_features_target,
    train_test_split_data,
)
from src.models import get_models


def main():
    raw_df, _, _ = load_raw_data()
    processed_df = prepare_churn_dataset(raw_df)

    X, y = split_features_target(processed_df)
    X_train, _, y_train, _ = train_test_split_data(X, y)

    model = get_models(X_train)["Gradient Boosting"]

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("Generating out-of-fold predictions...")

    probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_train,
            predictions,
            labels=[0, 1],
        ).ravel()

        results.append({
            "threshold": threshold,
            "precision": precision_score(
                y_train, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_train, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_train, predictions, zero_division=0
            ),
            "flagged_customers": int(predictions.sum()),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
        })

    results_df = pd.DataFrame(results)

    output_path = Path("results/threshold_cv_25features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_path, index=False)

    print("\nThreshold comparison:")
    print(results_df.round(4).to_string(index=False))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
