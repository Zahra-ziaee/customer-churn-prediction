from src.data_loader import load_raw_data
from src.preprocessing import (
    prepare_churn_dataset,
    save_processed_data,
    split_features_target,
    train_test_split_data,
)
from src.models import get_models, train_models
from src.evaluation import evaluate_all_models
from src.visualization import plot_churn_distribution, plot_model_comparison
from src.utils import save_results, print_best_model


def main():
    churn_df, zipcode_df, dictionary_df = load_raw_data()

    print("=" * 60)
    print("Customer Churn Prediction Project")
    print("=" * 60)

    print("\nRaw data shape:")
    print(churn_df.shape)

    processed_df = prepare_churn_dataset(churn_df)

    print("\nProcessed data shape:")
    print(processed_df.shape)

    print("\nTarget distribution:")
    print(processed_df["Churn"].value_counts())
    print(processed_df["Churn"].value_counts(normalize=True))

    save_processed_data(processed_df)
    plot_churn_distribution(processed_df)

    X, y = split_features_target(processed_df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    print("\nTrain/Test split:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    models = get_models(X_train)
    trained_models = train_models(models, X_train, y_train)

    results_df = evaluate_all_models(trained_models, X_test, y_test)

    print("\nModel comparison:")
    print(results_df)

    save_results(results_df)
    print_best_model(results_df)
    plot_model_comparison(results_df)


if __name__ == "__main__":
    main()