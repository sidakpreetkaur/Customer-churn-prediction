import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from streamlit_option_menu import option_menu


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("customer_churn_model.joblib")

# Load the exact column order the model was trained on (if available)
try:
    training_columns = joblib.load("training_columns.pkl")
except FileNotFoundError:
    training_columns = None


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("cleaned_churn.csv")


# ============================================================
# PAGE TITLE
# ============================================================

st.title("📊 Customer Churn Prediction Dashboard")
st.write("Analyze customer behavior and predict customer churn using Machine Learning.")


# ============================================================
# DATASET INFORMATION
# ============================================================

st.subheader("📋 Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Customers", df.shape[0])

with col2:
    st.metric("Total Features", df.shape[1])


# ============================================================
# CREATE CHURN LABEL
# ============================================================

# If your dataset contains Exited column
if "Exited" in df.columns:

    df["Churn Status"] = df["Exited"].apply(
        lambda x: "Churned" if x == 1 else "Active"
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    selected = option_menu(
        menu_title="📂 Navigation",

        options=[
            "Overview",
            "Customer Analysis",
            "Financial Analysis",
            "Churn Analysis",
            "ML Prediction"
        ],

        icons=[
            "house",
            "people",
            "cash-stack",
            "exclamation-triangle",
            "robot"
        ],

        default_index=0
    )


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.markdown("---")
st.sidebar.header("🔎 Filters")


# ------------------------------------------------------------
# Geography Filter
# ------------------------------------------------------------

if "Geography" in df.columns:

    geography = st.sidebar.selectbox(
        "Select Geography",
        ["All"] + sorted(df["Geography"].dropna().unique().tolist())
    )

else:

    geography = "All"


# ------------------------------------------------------------
# Gender Filter
# ------------------------------------------------------------

if "Gender" in df.columns:

    gender = st.sidebar.selectbox(
        "Select Gender",
        ["All"] + sorted(df["Gender"].dropna().unique().tolist())
    )

else:

    gender = "All"


# ------------------------------------------------------------
# Filter Data
# ------------------------------------------------------------

filtered_df = df.copy()


if geography != "All":
    filtered_df = filtered_df[
        filtered_df["Geography"] == geography
    ]


if gender != "All":
    filtered_df = filtered_df[
        filtered_df["Gender"] == gender
    ]


# ------------------------------------------------------------
# Developer
# ------------------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.markdown("### 👩‍💻 Developed by")

st.sidebar.markdown("**Sidakpreet Kaur**")


# ============================================================
# OVERVIEW
# ============================================================

if selected == "Overview":

    st.subheader("🔍 Current View Filters")

    st.markdown(
        f"""
        - 🌍 **Geography:** **{geography}**
        - 👤 **Gender:** **{gender}**
        """
    )


    # --------------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------------

    total_customers = len(filtered_df)

    if "Exited" in filtered_df.columns:

        churned_customers = (
            filtered_df["Exited"] == 1
        ).sum()

        active_customers = (
            filtered_df["Exited"] == 0
        ).sum()

        churn_rate = (
            churned_customers / total_customers * 100
            if total_customers > 0
            else 0
        )

    else:

        churned_customers = 0
        active_customers = 0
        churn_rate = 0


    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    st.subheader("📈 Key Performance Indicators")

    col1, col2, col3, col4, col5, col6 = st.columns(6)


    with col1:

        st.metric(
            "Total Customers",
            total_customers
        )


    with col2:

        st.metric(
            "Churned Customers",
            churned_customers
        )


    with col3:

        st.metric(
            "Active Customers",
            active_customers
        )


    with col4:

        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )


    with col5:

        if "CreditScore" in filtered_df.columns:

            st.metric(
                "Avg Credit Score",
                f"{filtered_df['CreditScore'].mean():.0f}"
            )

        else:

            st.metric(
                "Avg Credit Score",
                "N/A"
            )


    with col6:

        if "Balance" in filtered_df.columns:

            st.metric(
                "Avg Balance",
                f"${filtered_df['Balance'].mean():,.0f}"
            )

        else:

            st.metric(
                "Avg Balance",
                "N/A"
            )


    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    st.subheader("📋 Customer Data Preview")

    with st.expander(
        "View Customer Data",
        expanded=False
    ):

        st.dataframe(
            filtered_df.head(20),
            use_container_width=True
        )


# ============================================================
# CUSTOMER ANALYSIS
# ============================================================

elif selected == "Customer Analysis":

    st.title("👥 Customer Analysis")


    # --------------------------------------------------------
    # CHURN BY GEOGRAPHY
    # --------------------------------------------------------

    if "Geography" in filtered_df.columns and "Exited" in filtered_df.columns:

        st.subheader("🌍 Churn by Geography")

        geography_churn = (
            filtered_df
            .groupby("Geography")["Exited"]
            .mean()
            .reset_index()
        )

        geography_churn["Churn Rate"] = (
            geography_churn["Exited"] * 100
        )


        fig = px.bar(
            geography_churn,
            x="Geography",
            y="Churn Rate",
            title="Churn Rate by Geography",
            text="Churn Rate"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # CHURN BY GENDER
    # --------------------------------------------------------

    if "Gender" in filtered_df.columns and "Exited" in filtered_df.columns:

        st.subheader("👤 Churn by Gender")

        gender_churn = (
            filtered_df
            .groupby("Gender")["Exited"]
            .mean()
            .reset_index()
        )

        gender_churn["Churn Rate"] = (
            gender_churn["Exited"] * 100
        )


        fig = px.bar(
            gender_churn,
            x="Gender",
            y="Churn Rate",
            title="Churn Rate by Gender",
            text="Churn Rate"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # AGE DISTRIBUTION
    # --------------------------------------------------------

    if "Age" in filtered_df.columns:

        st.subheader("🎂 Customer Age Distribution")

        fig = px.histogram(
            filtered_df,
            x="Age",
            nbins=20,
            title="Customer Age Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # TENURE
    # --------------------------------------------------------

    if "Tenure" in filtered_df.columns:

        st.subheader("📅 Customers by Tenure")

        tenure_count = (
            filtered_df["Tenure"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        tenure_count.columns = [
            "Tenure",
            "Customers"
        ]


        fig = px.bar(
            tenure_count,
            x="Tenure",
            y="Customers",
            title="Customers by Tenure"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# FINANCIAL ANALYSIS
# ============================================================

elif selected == "Financial Analysis":

    st.title("💰 Financial Analysis")


    # --------------------------------------------------------
    # BALANCE DISTRIBUTION
    # --------------------------------------------------------

    if "Balance" in filtered_df.columns:

        st.subheader("💰 Balance Distribution")

        fig = px.histogram(
            filtered_df,
            x="Balance",
            nbins=30,
            title="Customer Balance Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # BALANCE VS CHURN
    # --------------------------------------------------------

    if (
        "Balance" in filtered_df.columns
        and "Exited" in filtered_df.columns
    ):

        st.subheader("💰 Balance vs Churn")

        fig = px.box(
            filtered_df,
            x="Exited",
            y="Balance",
            title="Balance Distribution by Churn Status"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # CREDIT SCORE VS CHURN
    # --------------------------------------------------------

    if (
        "CreditScore" in filtered_df.columns
        and "Exited" in filtered_df.columns
    ):

        st.subheader("📊 Credit Score vs Churn")

        fig = px.box(
            filtered_df,
            x="Exited",
            y="CreditScore",
            title="Credit Score by Churn Status"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # ESTIMATED SALARY
    # --------------------------------------------------------

    if "EstimatedSalary" in filtered_df.columns:

        st.subheader("💵 Estimated Salary Distribution")

        fig = px.histogram(
            filtered_df,
            x="EstimatedSalary",
            nbins=30,
            title="Estimated Salary Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# CHURN ANALYSIS
# ============================================================

elif selected == "Churn Analysis":

    st.title("⚠️ Customer Churn Analysis")


    # --------------------------------------------------------
    # CHURN VS ACTIVE
    # --------------------------------------------------------

    if "Exited" in filtered_df.columns:

        st.subheader("📊 Churned vs Active Customers")

        status_count = (
            filtered_df["Churn Status"]
            .value_counts()
            .reset_index()
        )

        status_count.columns = [
            "Status",
            "Customers"
        ]


        fig = px.pie(
            status_count,
            values="Customers",
            names="Status",
            hole=0.35,
            title="Churned vs Active Customers"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # CHURN BY GEOGRAPHY
    # --------------------------------------------------------

    if (
        "Geography" in filtered_df.columns
        and "Exited" in filtered_df.columns
    ):

        st.subheader("🌍 Churned Customers by Geography")

        churn_geo = filtered_df[
            filtered_df["Exited"] == 1
        ]

        churn_geo = (
            churn_geo["Geography"]
            .value_counts()
            .reset_index()
        )

        churn_geo.columns = [
            "Geography",
            "Churned Customers"
        ]


        fig = px.bar(
            churn_geo,
            x="Geography",
            y="Churned Customers",
            title="Churned Customers by Geography",
            text="Churned Customers"
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # CHURN BY GENDER
    # --------------------------------------------------------

    if (
        "Gender" in filtered_df.columns
        and "Exited" in filtered_df.columns
    ):

        st.subheader("👤 Churned Customers by Gender")

        churn_gender = filtered_df[
            filtered_df["Exited"] == 1
        ]

        churn_gender = (
            churn_gender["Gender"]
            .value_counts()
            .reset_index()
        )

        churn_gender.columns = [
            "Gender",
            "Churned Customers"
        ]


        fig = px.bar(
            churn_gender,
            x="Gender",
            y="Churned Customers",
            title="Churned Customers by Gender",
            text="Churned Customers"
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # ACTIVE MEMBER VS CHURN
    # --------------------------------------------------------

    if (
        "IsActiveMember" in filtered_df.columns
        and "Exited" in filtered_df.columns
    ):

        st.subheader("🏦 Active Membership vs Churn")

        active_member_churn = (
            filtered_df
            .groupby("IsActiveMember")["Exited"]
            .mean()
            .reset_index()
        )

        active_member_churn["Churn Rate"] = (
            active_member_churn["Exited"] * 100
        )

        active_member_churn["Membership"] = (
            active_member_churn["IsActiveMember"]
            .map({
                0: "Inactive",
                1: "Active"
            })
        )


        fig = px.bar(
            active_member_churn,
            x="Membership",
            y="Churn Rate",
            title="Churn Rate by Active Membership",
            text="Churn Rate"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# ML PREDICTION
# ============================================================

elif selected == "ML Prediction":

    st.title("🤖 Customer Churn Prediction")

    st.write(
        "Enter customer information below to predict whether the customer is likely to churn."
    )


    # ========================================================
    # INPUTS
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650
        )


        geography_input = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"]
        )


        gender_input = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )


        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )


        tenure = st.number_input(
            "Tenure",
            min_value=0,
            max_value=10,
            value=5
        )


        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=50000.0
        )


    with col2:

        num_of_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=4,
            value=1
        )


        has_cr_card = st.selectbox(
            "Has Credit Card?",
            ["Yes", "No"]
        )


        is_active_member = st.selectbox(
            "Is Active Member?",
            ["Yes", "No"]
        )


        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=50000.0
        )


        st.write("")
        st.write("")


        # ----------------------------------------------------
        # PREDICTION BUTTON
        # ----------------------------------------------------

        predict_button = st.button(
            "🔮 Predict Churn",
            use_container_width=True
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame({

            "CreditScore": [credit_score],

            "Geography": [geography_input],

            "Gender": [gender_input],

            "Age": [age],

            "Tenure": [tenure],

            "Balance": [balance],

            "NumOfProducts": [num_of_products],

            "HasCrCard": [
                1 if has_cr_card == "Yes" else 0
            ],

            "IsActiveMember": [
                1 if is_active_member == "Yes" else 0
            ],

            "EstimatedSalary": [estimated_salary]

        })


        # ----------------------------------------------------
        # ONE HOT ENCODING
        # ----------------------------------------------------

        input_data = pd.get_dummies(
            input_data,
            columns=[
                "Geography",
                "Gender"
            ],
            drop_first=True
        )


        # ----------------------------------------------------
        # MATCH MODEL FEATURES
        # ----------------------------------------------------

        try:

            # Align columns to match what the model was trained on
            if training_columns is not None:

                input_data = input_data.reindex(
                    columns=training_columns,
                    fill_value=0
                )

            elif hasattr(model, "feature_names_in_"):

                input_data = input_data.reindex(
                    columns=model.feature_names_in_,
                    fill_value=0
                )


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            prediction = model.predict(
                input_data
            )[0]


            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            if hasattr(model, "predict_proba"):

                probability = model.predict_proba(
                    input_data
                )[0][1]

            else:

                probability = None


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.markdown("---")

            st.subheader("🎯 Prediction Result")


            if prediction == 1:

                st.error(
                    "🚨 Customer is likely to CHURN"
                )

            else:

                st.success(
                    "✅ Customer is likely to STAY"
                )


            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            if probability is not None:

                st.metric(
                    "Churn Probability",
                    f"{probability * 100:.2f}%"
                )


                st.progress(
                    float(probability)
                )


                if probability >= 0.70:

                    st.error(
                        "⚠️ High Churn Risk"
                    )

                elif probability >= 0.40:

                    st.warning(
                        "⚠️ Medium Churn Risk"
                    )

                else:

                    st.success(
                        "✅ Low Churn Risk"
                    )


        except Exception as e:

            st.error(
                "Prediction error occurred."
            )

            st.write(e)