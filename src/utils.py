from pathlib import Path

import pandas as pd


def save_results(results_df: pd.DataFrame, output_path: str = "results/model_results.csv") -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_file, index=False)

    print(f"\nModel results saved to: {output_file}")


def print_best_model(results_df: pd.DataFrame) -> None:
    best_model = results_df.iloc[0]

    print("\n" + "=" * 60)
    print("Best Model")
    print("=" * 60)
    print(f"Model: {best_model['model']}")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print(f"Precision: {best_model['precision']:.4f}")
    print(f"Recall: {best_model['recall']:.4f}")
    print(f"F1-score: {best_model['f1']:.4f}")
    print(f"ROC-AUC: {best_model['roc_auc']:.4f}")


def get_feature_names_from_pipeline(model_pipeline) -> list:
    """
    Extract transformed feature names from a scikit-learn Pipeline
    containing a ColumnTransformer named 'preprocessor'.
    """
    preprocessor = model_pipeline.named_steps["preprocessor"]

    feature_names = []

    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "remainder":
            continue

        if transformer_name == "num":
            feature_names.extend(columns)

        elif transformer_name == "cat":
            onehot = transformer.named_steps["onehot"]
            encoded_names = onehot.get_feature_names_out(columns)
            feature_names.extend(encoded_names.tolist())

    return feature_names


def extract_feature_importance(model_pipeline, top_n: int = 25) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models such as
    GradientBoostingClassifier or RandomForestClassifier.
    """
    model = model_pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        raise ValueError("This model does not provide feature_importances_.")

    feature_names = get_feature_names_from_pipeline(model_pipeline)
    importances = model.feature_importances_

    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    feature_importance_df = feature_importance_df.sort_values(
        by="importance",
        ascending=False,
    ).head(top_n)

    return feature_importance_df


def save_feature_importance(
    feature_importance_df: pd.DataFrame,
    output_path: str = "results/feature_importance.csv",
) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    feature_importance_df.to_csv(output_file, index=False)

    print(f"\nFeature importance saved to: {output_file}")