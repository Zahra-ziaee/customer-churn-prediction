from typing import Dict

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X_test, y_test) -> Dict:
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        roc_auc = None

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }

    return results


def evaluate_all_models(trained_models, X_test, y_test) -> pd.DataFrame:
    rows = []

    for model_name, model in trained_models.items():
        print(f"\nEvaluating {model_name}...")
        results = evaluate_model(model, X_test, y_test)

        rows.append(
            {
                "model": model_name,
                "accuracy": results["accuracy"],
                "precision": results["precision"],
                "recall": results["recall"],
                "f1": results["f1"],
                "roc_auc": results["roc_auc"],
            }
        )

        print("\nClassification Report:")
        print(results["classification_report"])
        print("Confusion Matrix:")
        print(results["confusion_matrix"])

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(by="roc_auc", ascending=False)

    return results_df