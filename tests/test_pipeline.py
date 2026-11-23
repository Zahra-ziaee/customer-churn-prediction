
import unittest

import numpy as np

from src.data_loader import load_raw_data
from src.preprocessing import (
    prepare_churn_dataset,
    split_features_target,
    train_test_split_data,
)
from src.models import get_models


class TestChurnPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        raw_df, _, _ = load_raw_data()

        cls.processed_df = prepare_churn_dataset(
            raw_df
        )

        cls.X, cls.y = split_features_target(
            cls.processed_df
        )

        (
            cls.X_train,
            cls.X_test,
            cls.y_train,
            cls.y_test,
        ) = train_test_split_data(
            cls.X,
            cls.y,
        )

    def test_dataset_structure(self):
        """Verify dataset size and target structure."""

        self.assertEqual(
            self.processed_df.shape,
            (7043, 26),
        )

        self.assertEqual(
            self.X.shape[1],
            25,
        )

        self.assertEqual(
            len(self.X),
            len(self.y),
        )

        self.assertEqual(
            set(self.y.unique()),
            {0, 1},
        )

        self.assertEqual(
            int(self.y.sum()),
            1869,
        )

        self.assertEqual(
            len(self.X_train),
            5634,
        )

        self.assertEqual(
            len(self.X_test),
            1409,
        )

    def test_excluded_features(self):
        """Check that excluded columns are not predictors."""

        excluded_columns = {
            "Customer ID",
            "Customer Status",
            "Churn",
            "Churn Category",
            "Churn Reason",
            "City",
            "Zip Code",
            "Latitude",
            "Longitude",
            "Total Charges",
            "Total Refunds",
            "Total Extra Data Charges",
            "Total Long Distance Charges",
            "Total Revenue",
        }

        remaining = (
            excluded_columns
            & set(self.X.columns)
        )

        self.assertEqual(
            remaining,
            set(),
            f"Unexpected excluded features: {remaining}",
        )

    def test_model_prediction(self):
        """
        Verify that the pipeline fits and returns
        valid probabilities.

        Use only a small subset of the training data.
        This is a functionality test, not a performance
        evaluation.
        """

        model = get_models(
            self.X_train
        )["Gradient Boosting"]

        X_small = self.X_train.iloc[:400]
        y_small = self.y_train.iloc[:400]

        X_check = self.X_train.iloc[400:410]

        model.fit(
            X_small,
            y_small,
        )

        probabilities = model.predict_proba(
            X_check
        )

        self.assertEqual(
            probabilities.shape,
            (10, 2),
        )

        self.assertTrue(
            np.isfinite(probabilities).all()
        )

        self.assertTrue(
            ((probabilities >= 0) &
             (probabilities <= 1)).all()
        )

        self.assertTrue(
            np.allclose(
                probabilities.sum(axis=1),
                1.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
