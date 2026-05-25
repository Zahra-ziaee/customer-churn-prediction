import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.data_loader import load_raw_data
from src.preprocessing import (
    prepare_churn_dataset,
    split_features_target,
    train_test_split_data,
)
from src.models import get_models, train_models


@st.cache_resource
def train_gradient_boosting_model():
    churn_df, _, _ = load_raw_data()

    processed_df = prepare_churn_dataset(churn_df)

    X, y = split_features_target(processed_df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    models = get_models(X_train)
    trained_models = train_models(models, X_train, y_train)

    gradient_boosting_model = trained_models["Gradient Boosting"]

    return gradient_boosting_model, churn_df, processed_df


def build_customer_input(raw_df: pd.DataFrame) -> pd.DataFrame:
    sample_customer = raw_df.iloc[[0]].copy()

    st.subheader("Customer Risk Input")

    col1, col2 = st.columns(2)

    with col1:
        sample_customer.loc[:, "Age"] = st.slider(
            "Age",
            min_value=int(raw_df["Age"].min()),
            max_value=int(raw_df["Age"].max()),
            value=45,
        )

        sample_customer.loc[:, "Tenure in Months"] = st.slider(
            "Tenure in Months",
            min_value=int(raw_df["Tenure in Months"].min()),
            max_value=int(raw_df["Tenure in Months"].max()),
            value=12,
        )

        sample_customer.loc[:, "Monthly Charge"] = st.slider(
            "Monthly Charge",
            min_value=float(raw_df["Monthly Charge"].min()),
            max_value=float(raw_df["Monthly Charge"].max()),
            value=70.0,
        )

        sample_customer.loc[:, "Number of Referrals"] = st.slider(
            "Number of Referrals",
            min_value=int(raw_df["Number of Referrals"].min()),
            max_value=int(raw_df["Number of Referrals"].max()),
            value=0,
        )

    with col2:
        sample_customer.loc[:, "Contract"] = st.selectbox(
            "Contract",
            options=sorted(raw_df["Contract"].dropna().unique()),
            index=0,
        )

        sample_customer.loc[:, "Internet Type"] = st.selectbox(
            "Internet Type",
            options=sorted(raw_df["Internet Type"].fillna("No Internet Service").unique()),
            index=0,
        )

        sample_customer.loc[:, "Online Security"] = st.selectbox(
            "Online Security",
            options=sorted(raw_df["Online Security"].fillna("No Internet Service").unique()),
            index=0,
        )

        sample_customer.loc[:, "Premium Tech Support"] = st.selectbox(
            "Premium Tech Support",
            options=sorted(raw_df["Premium Tech Support"].fillna("No Internet Service").unique()),
            index=0,
        )

    return sample_customer


def main():
    st.set_page_config(
        page_title="Customer Churn Prediction",
        layout="wide",
    )

    st.title("📉 Customer Churn Prediction Dashboard")

    st.write(
        "A machine learning dashboard for predicting telecom customer churn "
        "using customer demographics, services, contract, billing, and revenue features."
    )

    model, raw_df, processed_df = train_gradient_boosting_model()

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    churn_rate = processed_df["Churn"].mean()

    col1.metric("Customers", f"{processed_df.shape[0]:,}")
    col2.metric("Features", f"{processed_df.shape[1] - 1}")
    col3.metric("Churn Rate", f"{churn_rate * 100:.2f}%")

    st.subheader("Model Summary")

    st.markdown(
        """
        **Best model:** Gradient Boosting  
        **Accuracy:** 0.8460  
        **Precision:** 0.7357  
        **Recall:** 0.6551  
        **F1-score:** 0.6931  
        **ROC-AUC:** 0.9115  
        """
    )

    st.subheader("Business Context")

    st.markdown(
        """
        The most important churn drivers identified by the model include:

        - Month-to-month contracts
        - Number of referrals
        - Age
        - Monthly charge
        - Tenure in months
        - Lack of online security
        - Lack of premium tech support
        """
    )

    st.divider()

    customer_input_raw = build_customer_input(raw_df)

    if st.button("Predict Churn Risk"):
        processed_input = prepare_churn_dataset(customer_input_raw)
        X_input = processed_input.drop(columns=["Churn"])

        churn_probability = model.predict_proba(X_input)[0][1]
        churn_prediction = model.predict(X_input)[0]

        st.subheader("Prediction Result")

        if churn_prediction == 1:
            st.error(f"High churn risk detected. Probability: {churn_probability:.2%}")
        else:
            st.success(f"Low churn risk detected. Probability: {churn_probability:.2%}")

        st.progress(float(churn_probability))

        st.write("Interpretation:")

        if churn_probability >= 0.7:
            st.write(
                "This customer should be prioritized for retention actions such as "
                "discounts, contract upgrade offers, or support outreach."
            )
        elif churn_probability >= 0.4:
            st.write(
                "This customer has moderate churn risk and may benefit from targeted engagement."
            )
        else:
            st.write(
                "This customer currently appears relatively stable based on the model."
            )

    st.divider()

    st.subheader("Generated Project Outputs")

    st.markdown(
        """
        The project generates:

        - `results/model_results.csv`
        - `results/feature_importance.csv`
        - `results/figures/churn_distribution.png`
        - `results/figures/model_comparison_roc_auc.png`
        - `results/figures/feature_importance_gradient_boosting.png`
        """
    )


if __name__ == "__main__":
    main()