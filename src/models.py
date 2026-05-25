from typing import Dict

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.config import RANDOM_STATE


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def get_models(X) -> Dict[str, Pipeline]:
    preprocessor = build_preprocessor(X)

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_split=10,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    return models


def train_models(models, X_train, y_train):
    trained_models = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        print(f"{model_name} training finished.")

    return trained_models