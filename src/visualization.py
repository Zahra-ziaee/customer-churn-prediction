from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR


def plot_model_comparison(results_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(results_df["model"], results_df[metric])
        plt.title(f"Model Comparison - {metric.upper()}")
        plt.ylabel(metric.upper())
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        output_path = FIGURES_DIR / f"model_comparison_{metric}.png"
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Saved chart: {output_path}")


def plot_churn_distribution(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    churn_counts = df["Churn"].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    plt.bar(["Not Churned", "Churned"], churn_counts.values)
    plt.title("Customer Churn Distribution")
    plt.ylabel("Number of Customers")
    plt.tight_layout()

    output_path = FIGURES_DIR / "churn_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved chart: {output_path}")