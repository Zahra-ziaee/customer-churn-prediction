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