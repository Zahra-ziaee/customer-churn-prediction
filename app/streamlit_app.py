
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# =====================================================
# PROJECT CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_raw_data
from src.preprocessing import (
    prepare_churn_dataset,
    split_features_target,
    train_test_split_data,
)
from src.models import get_models


SCREENING_THRESHOLD = 0.30

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


# =====================================================
# MODEL TRAINING
# =====================================================

@st.cache_resource
def load_and_train_model():
    """
    Train the Gradient Boosting model using 25 features.

    The model is fitted only on the training partition.
    """

    raw_df, _, _ = load_raw_data()

    processed_df = prepare_churn_dataset(raw_df)

    X, y = split_features_target(processed_df)

    X_train, _, y_train, _ = train_test_split_data(
        X, y
    )

    model = get_models(X_train)["Gradient Boosting"]
    model.fit(X_train, y_train)

    return model, processed_df, X.columns.tolist()


# =====================================================
# CUSTOMER INPUT HELPERS
# =====================================================

def internet_addon_input(label, has_internet):

    if not has_internet:
        st.text_input(
            label,
            value="No Internet Service",
            disabled=True,
        )
        return "No Internet Service"

    return st.selectbox(
        label,
        options=["No", "Yes"],
    )


def build_customer_input(feature_columns):
    """
    Collect all 25 features explicitly.

    No customer records are copied from the dataset.
    """

    st.subheader("Customer Risk Input")

    st.caption(
        "All 25 model features are specified below. "
        "Values for unavailable services are "
        "assigned automatically."
    )

    values = {}

    # -------------------------------------------------
    # CUSTOMER PROFILE
    # -------------------------------------------------

    st.markdown("#### Customer Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        values["Gender"] = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

        values["Age"] = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=45,
            step=1,
        )

    with col2:
        values["Married"] = st.selectbox(
            "Married",
            ["No", "Yes"],
        )

        values["Number of Dependents"] = (
            st.number_input(
                "Number of Dependents",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
            )
        )

    with col3:
        values["Number of Referrals"] = (
            st.number_input(
                "Number of Referrals",
                min_value=0,
                max_value=100,
                value=0,
                step=1,
            )
        )

        values["Tenure in Months"] = (
            st.number_input(
                "Tenure in Months",
                min_value=0,
                max_value=72,
                value=12,
                step=1,
            )
        )

    # -------------------------------------------------
    # CONTRACT AND BILLING
    # -------------------------------------------------

    st.markdown("#### Contract and Billing")

    col1, col2, col3 = st.columns(3)

    with col1:
        values["Contract"] = st.selectbox(
            "Contract",
            ["Month-to-Month", "One Year", "Two Year"],
        )

        values["Offer"] = st.selectbox(
            "Offer",
            [
                "No Offer",
                "Offer A",
                "Offer B",
                "Offer C",
                "Offer D",
                "Offer E",
            ],
        )

    with col2:
        values["Paperless Billing"] = st.selectbox(
            "Paperless Billing",
            ["No", "Yes"],
        )

        values["Payment Method"] = st.selectbox(
            "Payment Method",
            [
                "Bank Withdrawal",
                "Credit Card",
                "Mailed Check",
            ],
        )

    with col3:
        values["Monthly Charge"] = st.number_input(
            "Monthly Charge",
            min_value=0.0,
            max_value=1000.0,
            value=70.0,
            step=5.0,
            format="%.2f",
        )

    # -------------------------------------------------
    # PHONE SERVICE
    # -------------------------------------------------

    st.markdown("#### Phone Service")

    col1, col2, col3 = st.columns(3)

    with col1:
        values["Phone Service"] = st.selectbox(
            "Phone Service",
            ["Yes", "No"],
        )

    has_phone = values["Phone Service"] == "Yes"

    with col2:
        if has_phone:
            values["Multiple Lines"] = st.selectbox(
                "Multiple Lines",
                ["No", "Yes"],
            )
        else:
            st.text_input(
                "Multiple Lines",
                value="No Phone Service",
                disabled=True,
            )
            values["Multiple Lines"] = "No Phone Service"

    with col3:
        if has_phone:
            values[
                "Avg Monthly Long Distance Charges"
            ] = st.number_input(
                "Avg Monthly Long Distance Charges",
                min_value=0.0,
                max_value=1000.0,
                value=10.0,
                step=1.0,
                format="%.2f",
            )
        else:
            st.number_input(
                "Avg Monthly Long Distance Charges",
                min_value=0.0,
                value=0.0,
                disabled=True,
            )
            values[
                "Avg Monthly Long Distance Charges"
            ] = 0.0

    # -------------------------------------------------
    # INTERNET SERVICE
    # -------------------------------------------------

    st.markdown("#### Internet Service")

    col1, col2, col3 = st.columns(3)

    with col1:
        values["Internet Service"] = st.selectbox(
            "Internet Service",
            ["Yes", "No"],
        )

    has_internet = (
        values["Internet Service"] == "Yes"
    )

    with col2:
        if has_internet:
            values["Internet Type"] = st.selectbox(
                "Internet Type",
                ["Cable", "DSL", "Fiber Optic"],
            )
        else:
            st.text_input(
                "Internet Type",
                value="No Internet Service",
                disabled=True,
            )
            values["Internet Type"] = (
                "No Internet Service"
            )

    with col3:
        if has_internet:
            values["Avg Monthly GB Download"] = (
                st.number_input(
                    "Avg Monthly GB Download",
                    min_value=0.0,
                    max_value=1000.0,
                    value=20.0,
                    step=1.0,
                    format="%.2f",
                )
            )
        else:
            st.number_input(
                "Avg Monthly GB Download",
                min_value=0.0,
                value=0.0,
                disabled=True,
            )
            values["Avg Monthly GB Download"] = 0.0

    # -------------------------------------------------
    # INTERNET ADD-ON SERVICES
    # -------------------------------------------------

    st.markdown("#### Internet Add-on Services")

    addons = [
        "Online Security",
        "Online Backup",
        "Device Protection Plan",
        "Premium Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Streaming Music",
        "Unlimited Data",
    ]

    addon_col1, addon_col2 = st.columns(2)

    for index, feature in enumerate(addons):

        target_col = (
            addon_col1
            if index % 2 == 0
            else addon_col2
        )

        with target_col:
            values[feature] = internet_addon_input(
                feature,
                has_internet,
            )

    # -------------------------------------------------
    # INPUT VALIDATION
    # -------------------------------------------------

    expected = set(feature_columns)
    provided = set(values)

    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)

    if missing or unexpected:
        raise ValueError(
            f"Feature mismatch. "
            f"Missing: {missing}. "
            f"Unexpected: {unexpected}."
        )

    customer_df = pd.DataFrame([values])

    customer_df = customer_df.loc[
        :, feature_columns
    ]

    if customer_df.isna().any().any():
        raise ValueError(
            "Customer input contains missing values."
        )

    return customer_df


# =====================================================
# MODEL SUMMARY
# =====================================================

def show_model_summary():

    st.subheader("Model Summary")

    cv_file = RESULTS_DIR / "model_cv_25features.csv"

    if not cv_file.exists():
        st.warning(
            "Cross-validation results unavailable. "
            "Run python compare_models_cv.py first."
        )
        return

    cv_df = pd.read_csv(cv_file)

    selected_rows = cv_df.loc[
        cv_df["model"] == "Gradient Boosting"
    ]

    if selected_rows.empty:
        st.warning(
            "Gradient Boosting not found in CV results."
        )
        return

    selected = selected_rows.iloc[0]

    st.write("**Selected model: Gradient Boosting**")

    st.caption(
        "Three-fold cross-validation on the training "
        "partition of the 25-feature dataset. "
        "These are not independent test-set results."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "ROC-AUC (CV)",
        f"{selected['roc_auc_mean']:.4f}",
    )

    col2.metric(
        "F1-score (CV)",
        f"{selected['f1_mean']:.4f}",
    )

    col3.metric(
        "Precision (CV)",
        f"{selected['precision_mean']:.4f}",
    )

    col4.metric(
        "Recall (CV)",
        f"{selected['recall_mean']:.4f}",
    )

    with st.expander(
        "View cross-validation model comparison"
    ):
        st.dataframe(
            cv_df.round(4),
            use_container_width=True,
            hide_index=True,
        )


# =====================================================
# PERMUTATION FEATURE IMPORTANCE
# =====================================================

def show_feature_importance():

    st.divider()

    st.subheader("Permutation Feature Importance")

    st.write(
        "Importance is measured as the decrease "
        "in validation ROC-AUC after randomly "
        "shuffling one feature at a time."
    )

    chart_file = (
        FIGURES_DIR
        / "feature_importance_25features_permutation.png"
    )

    importance_file = (
        RESULTS_DIR
        / "feature_importance_25features_permutation.csv"
    )

    if chart_file.exists():
        st.image(
            str(chart_file),
            caption=(
                "Top 12 features ranked by mean decrease "
                "in validation ROC-AUC. Error bars show "
                "variation across five permutations."
            ),
            use_container_width=True,
        )
    else:
        st.warning(
            "Feature importance chart missing. "
            "Run python plot_feature_importance_25.py first."
        )

    if importance_file.exists():

        importance_df = pd.read_csv(
            importance_file
        ).sort_values(
            "importance_mean",
            ascending=False,
        )

        if not importance_df.empty:
            top_feature = importance_df.iloc[0]

            st.info(
                f"Model Insight: {top_feature['feature']} "
                "has the highest measured permutation "
                "importance in this validation analysis. "
                "This does not establish causation."
            )

        with st.expander(
            "View all 25 feature importance values"
        ):
            st.dataframe(
                importance_df.round(5),
                use_container_width=True,
                hide_index=True,
            )

    else:
        st.warning(
            "Feature importance results missing. "
            "Run python analyze_feature_importance_25.py first."
        )

    st.caption(
        "Permutation importance was calculated on an "
        "internal validation split from the training "
        "partition. The explanation model was trained "
        "on an inner training split, not the complete "
        "dashboard training partition. Correlated "
        "features may share importance. "
        "These results do not establish causation."
    )


# =====================================================
# CUSTOMER SEGMENTATION
# =====================================================

def show_customer_segmentation():

    st.divider()

    st.subheader("Customer Churn Segmentation")

    st.write(
        "Observed churn rates by contract type "
        "and customer tenure."
    )

    heatmap_file = (
        FIGURES_DIR
        / "churn_rate_by_contract_tenure.png"
    )

    if heatmap_file.exists():
        st.image(
            str(heatmap_file),
            caption=(
                "Observed churn rates and churn counts "
                "by contract and tenure group."
            ),
            use_container_width=True,
        )
    else:
        st.warning(
            "Heatmap missing. "
            "Run python plot_churn_segments.py first."
        )

    st.info(
        "Business Insight: Month-to-month customers "
        "with 0-12 months of tenure accounted for "
        "approximately 54.8% of observed churn cases. "
        "This segment may be worth investigating, "
        "but this does not prove that changing a "
        "contract would prevent churn."
    )

    segment_file = (
        RESULTS_DIR
        / "churn_segmentation_by_tenure_contract.csv"
    )

    if segment_file.exists():

        with st.expander(
            "View detailed customer segmentation data"
        ):

            segments = pd.read_csv(
                segment_file
            ).copy()

            segments["churn_rate"] = (
                segments["churn_rate"] * 100
            ).round(2)

            segments["share_of_all_churn"] = (
                segments["share_of_all_churn"] * 100
            ).round(2)

            segments = segments.rename(
                columns={
                    "customers": "Customers",
                    "churned": "Churned",
                    "churn_rate": "Churn Rate (%)",
                    "share_of_all_churn": (
                        "Share of All Churn (%)"
                    ),
                }
            )

            st.dataframe(
                segments,
                use_container_width=True,
                hide_index=True,
            )


# =====================================================
# BUSINESS CONTEXT
# =====================================================

def show_business_analysis():

    st.subheader("Business Context")

    st.write(
        "This project combines predictive modeling "
        "with customer segmentation and hypothetical "
        "retention economics."
    )

    st.markdown(
        """
        Key analytical questions:

        - Which customer segments have higher churn rates?
        - Which features affect validation performance?
        - How does the threshold change precision and recall?
        - How might outreach costs affect retention decisions?

        The dataset is observational. Associations
        do not prove causal relationships.
        """
    )

    show_feature_importance()
    show_customer_segmentation()


# =====================================================
# BUSINESS DECISION ANALYSIS
# =====================================================

def show_business_decision_analysis():
    """
    Display threshold performance and hypothetical
    business impact scenarios.
    """

    st.divider()

    st.subheader("Business Decision Analysis")

    st.write(
        "Compare screening thresholds using "
        "out-of-fold classification performance "
        "and hypothetical retention economics."
    )

    # -------------------------------------------------
    # BUSINESS IMPACT CHART
    # -------------------------------------------------

    chart_file = (
        FIGURES_DIR
        / "business_impact_by_threshold.png"
    )

    if chart_file.exists():
        st.image(
            str(chart_file),
            caption=(
                "Estimated net value under two "
                "hypothetical outreach-cost scenarios."
            ),
            use_container_width=True,
        )
    else:
        st.warning(
            "Business impact chart not found. "
            "Run python plot_business_impact.py first."
        )

    # -------------------------------------------------
    # BUSINESS IMPACT TABLE
    # -------------------------------------------------

    business_file = (
        RESULTS_DIR
        / "business_impact_scenarios.csv"
    )

    if business_file.exists():

        business_df = pd.read_csv(business_file)

        if not business_df.empty:

            st.markdown(
                "#### Best Observed Threshold by Scenario"
            )

            best_rows = (
                business_df.sort_values(
                    "estimated_net_value",
                    ascending=False,
                )
                .drop_duplicates("scenario")
                .sort_values("scenario")
            )

            columns = st.columns(len(best_rows))

            for column, (_, row) in zip(
                columns,
                best_rows.iterrows(),
            ):

                with column:
                    st.metric(
                        row["scenario"],
                        f"{row['threshold']:.0%}",
                    )

                    st.caption(
                        "Estimated net value: "
                        f"{row['estimated_net_value']:,.0f} "
                        "hypothetical units"
                    )

        with st.expander(
            "View hypothetical business impact scenarios"
        ):

            display_df = business_df.copy()

            display_df["threshold"] = (
                display_df["threshold"] * 100
            ).round(0)

            display_df = display_df.rename(
                columns={
                    "threshold": "Threshold (%)"
                }
            )

            st.dataframe(
                display_df.round(2),
                use_container_width=True,
                hide_index=True,
            )

    else:
        st.warning(
            "Business impact data unavailable. "
            "Run python analyze_business_impact.py first."
        )

    # -------------------------------------------------
    # THRESHOLD COMPARISON
    # -------------------------------------------------

    threshold_file = (
        RESULTS_DIR
        / "threshold_cv_25features.csv"
    )

    if threshold_file.exists():

        with st.expander(
            "View threshold performance comparison"
        ):

            threshold_df = pd.read_csv(
                threshold_file
            ).copy()

            for metric in [
                "precision",
                "recall",
                "f1",
            ]:
                threshold_df[metric] = (
                    threshold_df[metric] * 100
                ).round(2)

            threshold_df["threshold"] = (
                threshold_df["threshold"] * 100
            ).round(0)

            threshold_df = threshold_df.rename(
                columns={
                    "threshold": "Threshold (%)",
                    "precision": "Precision (%)",
                    "recall": "Recall (%)",
                    "f1": "F1 (%)",
                }
            )

            st.dataframe(
                threshold_df,
                use_container_width=True,
                hide_index=True,
            )

    else:
        st.warning(
            "Threshold results unavailable. "
            "Run python analyze_thresholds_cv.py first."
        )

    st.info(
        "Decision Insight: Among the tested thresholds, "
        "30% produced the highest estimated net value "
        "in the low-cost scenario, while 50% performed "
        "best in the high-cost scenario. "
        "These are not measured campaign profits."
    )

    st.caption(
        "Threshold comparisons use out-of-fold predictions "
        "from the training partition. Outreach costs, "
        "retention success rates, and customer values are "
        "hypothetical. The analysis does not establish "
        "causal retention effects or independently "
        "validated future performance."
    )


# =====================================================
# PREDICTION RESULT
# =====================================================

def show_prediction_result(model, customer_df):

    probability = float(
        model.predict_proba(customer_df)[0, 1]
    )

    flagged = (
        probability >= SCREENING_THRESHOLD
    )

    st.subheader("Prediction Result")

    st.metric(
        "Estimated Churn Probability",
        f"{probability:.2%}",
    )

    st.progress(probability)

    if flagged:

        st.warning(
            "Flagged for retention review "
            f"(screening threshold: "
            f"{SCREENING_THRESHOLD:.0%})."
        )

        st.write(
            "This customer may be considered "
            "for further retention analysis."
        )

    else:

        st.success(
            "Not flagged for retention review "
            f"at the {SCREENING_THRESHOLD:.0%} "
            "screening threshold."
        )

        st.write(
            "A score below the illustrative threshold "
            "does not guarantee that the customer will stay."
        )

    st.caption(
        "The 30% screening threshold is illustrative "
        "and was selected using training-only "
        "cross-validation and hypothetical business "
        "assumptions. Model probabilities are not "
        "independently calibrated, and prospective "
        "performance has not been validated."
    )

    with st.expander(
        "View all 25 features used for this prediction"
    ):

        st.dataframe(
            customer_df.T.rename(
                columns={0: "Input Value"}
            ),
            use_container_width=True,
        )


# =====================================================
# GENERATED PROJECT OUTPUTS
# =====================================================

def show_project_outputs():

    st.divider()

    st.subheader("Generated Project Outputs")

    st.markdown(
        """
        **Model evaluation**

        - `results/model_results_baseline_32features.csv`
        - `results/model_cv_25features.csv`
        - `results/threshold_cv_25features.csv`

        **Business analysis**

        - `results/business_impact_scenarios.csv`
        - `results/churn_segmentation_by_tenure_contract.csv`
        - `results/feature_importance_25features_permutation.csv`

        **Visualizations**

        - `results/figures/churn_distribution.png`
        - `results/figures/model_comparison_roc_auc.png`
        - `results/figures/feature_importance_25features_permutation.png`
        - `results/figures/churn_rate_by_contract_tenure.png`
        - `results/figures/business_impact_by_threshold.png`
        """
    )


# =====================================================
# MAIN DASHBOARD
# =====================================================

def main():

    st.set_page_config(
        page_title="Customer Churn Prediction",
        page_icon="📉",
        layout="wide",
    )

    st.title(
        "📉 Customer Churn Prediction Dashboard"
    )

    st.write(
        "A machine learning and business analytics "
        "dashboard for exploring telecom churn, "
        "comparing models, interpreting feature "
        "importance, and analyzing retention decisions."
    )

    model, processed_df, feature_columns = (
        load_and_train_model()
    )

    # Dataset overview
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Customers",
        f"{len(processed_df):,}",
    )

    col2.metric(
        "Model Features",
        str(len(feature_columns)),
    )

    col3.metric(
        "Churn Rate",
        f"{processed_df['Churn'].mean():.2%}",
    )

    # Analytical sections
    show_model_summary()

    show_business_analysis()

    show_business_decision_analysis()

    # Customer prediction
    st.divider()

    customer_df = build_customer_input(
        feature_columns
    )

    if st.button(
        "Predict Churn Risk",
        type="primary",
    ):

        show_prediction_result(
            model,
            customer_df,
        )

    # Generated outputs
    show_project_outputs()


if __name__ == "__main__":
    main()
