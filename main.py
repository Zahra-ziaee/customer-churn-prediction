
from pathlib import Path

import pandas as pd

from src.data_loader import load_raw_data
from src.preprocessing import (
    prepare_churn_dataset,
    save_processed_data,
    split_features_target,
    train_test_split_data,
)
from src.visualization import plot_churn_distribution
from compare_models_cv import main as run_cross_validation


PROJECT_ROOT = Path(__file__).resolve().parent

CV_RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "model_cv_25features.csv"
)


def main():
    print("=" * 60)
    print("Customer Churn Prediction")
    print("Business Machine Learning Analysis")
    print("=" * 60)

    # Load original data.
    raw_df, _, _ = load_raw_data()

    print(f"\nRaw data shape: {raw_df.shape}")

    # Prepare the reduced modeling dataset.
    processed_df = prepare_churn_dataset(raw_df)

    X, y = split_features_target(processed_df)

    print(f"Processed data shape: {processed_df.shape}")
    print(f"Model features: {X.shape[1]}")

    if X.shape[1] != 25:
        raise ValueError(
            f"Expected 25 model features, got {X.shape[1]}."
        )

    print(f"Overall churn rate: {y.mean():.2%}")

    # Save processed data and descriptive chart.
    save_processed_data(processed_df)
    plot_churn_distribution(processed_df)

    # Confirm the train/test partition.
    X_train, X_test, y_train, y_test = (
        train_test_split_data(X, y)
    )

    print("\nTrain/Test split:")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    print(
        "\nThe existing test partition is not used "
        "for model selection."
    )

    # Compare models using training-only CV.
    run_cross_validation()

    if not CV_RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"CV results not found: {CV_RESULTS_FILE}"
        )

    cv_results = pd.read_csv(CV_RESULTS_FILE)

    cv_results = cv_results.sort_values(
        by="roc_auc_mean",
        ascending=False,
    )

    best = cv_results.iloc[0]

    print("\n" + "=" * 60)
    print("Best Model Based on Training-Only CV")
    print("=" * 60)

    print(f"Model: {best['model']}")
    print(f"ROC-AUC: {best['roc_auc_mean']:.4f}")
    print(f"F1-score: {best['f1_mean']:.4f}")
    print(f"Precision: {best['precision_mean']:.4f}")
    print(f"Recall: {best['recall_mean']:.4f}")

    if best["model"] != "Gradient Boosting":
        print(
            "\nWARNING: The highest-ranked CV model "
            "differs from the Gradient Boosting model "
            "currently used by the Streamlit dashboard."
        )

    print("\nAnalysis complete.")
    print(
        "Other analytical scripts generate threshold, "
        "segmentation, feature importance, and "
        "hypothetical business impact reports."
    )

    print(
        "\nNote: This is a training-only model "
        "comparison, not a new independent "
        "test-set evaluation."
    )


if __name__ == "__main__":
    main()
