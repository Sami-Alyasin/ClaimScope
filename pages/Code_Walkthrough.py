import streamlit as st

# ------------------------
# 1. Page Configuration
# ------------------------

st.set_page_config(
    page_title="Code Walkthrough - ClaimScope",
    page_icon="🧑‍💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("<h1 style='text-align: center;'> 🧑‍💻 Code Walkthrough</h1>", unsafe_allow_html=True)    


# st.title("🧑‍💻 Code Walkthrough")

# Data Generation
with st.container(border=True):
    st.header("Data Generation 🛠️", divider=True)
    st.subheader("`01_data_generation.py`")
    st.markdown("""
    This script generates a synthetic insurance dataset with correlated features, noise, outliers, time dependencies, and missing values. The key steps are:
    - **Generate Policyholder Data**: Creates policyholders with features like age, gender, location, and location risk score.
    - **Generate Claims**: Creates claims for each policyholder, embedding time dependencies and correlations.
    - **Introduce Fraud**: Labels a subset of claims as fraud, injecting patterns like exaggerated claim amounts and suspicious keywords.
    - **Introduce Noise and Outliers**: Adds random noise and outliers in claim amounts and dates.
    - **Introduce Missing Values**: Randomly replaces a fraction of the dataframe cells with NaN to simulate real-world data.

    The output is saved as `synthetic_claims.csv`.
    """)

# Exploratory Data Analysis
with st.container(border=True):
    st.header("Exploratory Data Analysis 📊", divider=True)
    st.subheader("`02_EDA.ipynb`")
    st.markdown("""
    This Jupyter notebook performs exploratory data analysis on the synthetic dataset. Key steps include:
    - **Loading the Dataset**: Reads the `synthetic_claims.csv` file.
    - **Basic Overview**: Provides a summary of the dataset, including data types and missing values.
    - **Distribution Analysis**: Plots histograms for key numeric columns and a pie chart for fraud percentage.
    - **Correlation Analysis**: Generates a heatmap to show correlations between numeric features.
    - **Categorical Analysis**: Analyzes the distribution of categorical features like claim type.
    """)

# Data Preprocessing
with st.container(border=True):
    st.header("Data Preprocessing 🧹", divider=True)
    st.subheader("`03_preprocessing.py`")
    st.markdown("""
    This script performs basic data cleaning and preprocessing on `synthetic_claims.csv`. Key steps include:
    - **Remove Duplicates**: Removes exact duplicate rows.
    - **Drop Rows with Missing Dates**: Drops rows that are missing both `incident_date` and `report_date`.
    - **Drop Negative Claim Amounts**: Removes rows with negative claim amounts.
    - **Save Preprocessed Data**: Saves the cleaned dataset as `preprocessed_claims.csv`.
    """)

# Feature Engineering
with st.container(border=True):
    st.header("Feature Engineering 🔧", divider=True)
    st.subheader("`04_feature_engineering.py`")
    st.markdown("""
    This script creates new features, handles missing data, and outputs a cleaned/processed CSV with the new features. Key steps include:
    - **Create New Features**: Adds features like `report_delay_days`, `suspicious_note_flag`, and `location_type`.
    - **Handle Missing Values**: Imputes missing values for numeric and categorical columns.
    - **Save Processed Data**: Saves the processed dataset as `processed_claims.csv`.
    """)

# Modeling
with st.container(border=True):
    st.header("Modeling 🤖", divider=True)
    st.subheader("`05_modeling.py`")
    st.markdown("""
    This script trains classification models for fraud detection on `processed_claims.csv`, evaluates performance, and saves model artifacts. Key steps include:
    - **Load Processed Data**: Reads the `processed_claims.csv` file.
    - **Define Features and Target**: Defines the features and target variable for modeling.
    - **Train/Test Split**: Splits the data into training and test sets.
    - **Scale Numeric Features**: Scales continuous numeric features.
    - **One-Hot Encode Categorical Features**: Encodes categorical features.
    - **Train Models**: Trains logistic regression and random forest models.
    - **Evaluate Models**: Evaluates the models using metrics like confusion matrix, classification report, and ROC-AUC score.
    - **Save Model Artifacts**: Saves the best model and scaler as `fraud_detector_model.pkl` and `scaler.pkl`.
    - **Generate Feature Importance Visual**: Creates a bar plot of feature importances and saves it as `feature_importance.png`.

    Hopefully this walkthrough helps you understand the codebase and the steps involved in building the ClaimScope project.
    """)
st.markdown("---")
st.markdown("""
**ClaimScope** | Developed by [Sami Alyasin](#) | [GitHub Repository](https://github.com/Sami-Alyasin/ClaimScope)
""")