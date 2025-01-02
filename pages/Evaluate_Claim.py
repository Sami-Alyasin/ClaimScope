# app/submit_claim.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os
import io
import requests
import joblib

# URLs for the model and scaler
model_url = "https://raw.githubusercontent.com/Sami-Alyasin/ClaimScope/refs/heads/main/models/fraud_detector_model.pkl"
scaler_url = "https://raw.githubusercontent.com/Sami-Alyasin/ClaimScope/refs/heads/main/models/scaler.pkl"

# Function to download and load the model
def load_model(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    return joblib.load(io.BytesIO(response.content))

# Load the model and scaler
model = load_model(model_url)
scaler = load_model(scaler_url)


def load_model(model_path):
    """
    Loads the trained Random Forest model from the specified path.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def load_scaler(scaler_path):
    """
    Loads the StandardScaler from the specified path.
    """
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler

def compute_location_risk_score(location_type, age):
    """
    Computes the location_risk_score based on location type and age.
    """
    if location_type == "Urban":
        risk_score_base = 0.8
    elif location_type == "Suburban":
        risk_score_base = 0.5
    elif location_type == "Rural":
        risk_score_base = 0.3
    else:
        risk_score_base = 0.0  # Default if location_type is unknown

    location_risk_score = risk_score_base + (age - 45) * 0.001
    location_risk_score = max(0.0, min(location_risk_score, 1.5))
    return location_risk_score

def compute_report_delay_days(claim_date, report_date):
    """
    Computes the number of days between claim date and report date.
    """
    delta = report_date - claim_date
    return max(delta.days, 0)  # Ensure non-negative

def compute_suspicious_note_flag(adjuster_notes):
    """
    Computes the suspicious_note_flag based on adjuster_notes.
    """

    suspicious_terms = [
        # Initial terms
        "inconsistent statements", "fabricated evidence", "urgent payout demanded",
        
        # Documentation issues
        "missing documentation", "altered documents", "backdated", "unclear documentation",
        "conflicting reports", "unsigned documents", "incomplete forms", "multiple versions",
        
        # Behavior patterns
        "aggressive demands", "threatening language", "refuses to cooperate", "avoids contact",
        "changes story", "excessive complaints", "unreachable claimant", "repeat claims",
        
        # Timing and circumstances
        "late reporting", "convenient timing", "recent policy change", "policy about to lapse",
        "weekend incident", "no witnesses", "suspicious timing", "multiple policies",
        
        # Financial indicators
        "financial difficulties", "bankruptcy filing", "debt mentioned", "overvalued claim",
        "multiple claims history", "previous denials", "inflated values", "cash only",
        
        # Medical red flags
        "pre existing condition", "doctor shopping", "minimal damage", "excessive treatment",
        "soft tissue only", "delayed symptoms", "multiple providers", "refuses examination",
        
        # Vehicle/Property specific
        "prior damage", "mechanical failure", "maintenance issues", "wear and tear",
        "unknown location", "unwitnessed theft", "suspicious fire", "vacant property",
        
        # Communication patterns
        "third party pressure", "lawyer immediate", "rehearsed statement", "scripted answers",
        "defensive responses", "evasive answers", "contradictory info", "refuses details"
    ]
    adjuster_notes = str(adjuster_notes).lower()
    return int(any(term in adjuster_notes for term in suspicious_terms))

def one_hot_encode_feature(df, feature, categories):
    """
    One-hot encodes a single feature based on the provided categories.
    """
    for category in categories:
        column_name = f"{feature}_{category.lower()}"
        df[column_name] = int(df[feature].str.lower() == category.lower())
    return df

def preprocess_input(user_input, model_features):
    """
    Processes user input and prepares the feature vector for the model.
    """
    # 1. Compute location_risk_score
    location_risk_score = compute_location_risk_score(user_input['location_type'], user_input['age'])

    # 2. Compute report_delay_days
    report_delay_days = compute_report_delay_days(user_input['claim_date'], user_input['report_date'])

    # 3. Compute suspicious_note_flag
    suspicious_note_flag = compute_suspicious_note_flag(user_input['adjuster_notes'])

    # 4. One-Hot Encode categorical features
    df = pd.DataFrame([user_input])

    # One-Hot Encode claim_type
    claim_type_categories = ['Auto', 'Health', 'Liability', 'Property']
    df = one_hot_encode_feature(df, 'claim_type', claim_type_categories)

    # One-Hot Encode gender
    gender_categories = ['F', 'M', 'Unknown']
    df = one_hot_encode_feature(df, 'gender', gender_categories)

    # One-Hot Encode location_type
    location_type_categories = ['Rural', 'Urban', 'Suburban']
    df = one_hot_encode_feature(df, 'location_type', location_type_categories)

    # 5. Prepare the final feature DataFrame
    features = {
        'claim_amount': user_input['claim_amount'],
        'age': user_input['age'],
        'location_risk_score': location_risk_score,
        'report_delay_days': report_delay_days,
        'suspicious_note_flag': suspicious_note_flag,
        'claim_type_auto': df.get('claim_type_auto', 0),
        'claim_type_health': df.get('claim_type_health', 0),
        'claim_type_liability': df.get('claim_type_liability', 0),
        'claim_type_property': df.get('claim_type_property', 0),
        'gender_F': df.get('gender_f', 0),
        'gender_M': df.get('gender_m', 0),
        'gender_unknown': df.get('gender_unknown', 0),
        'location_type_rural': df.get('location_type_rural', 0),
        'location_type_urban': df.get('location_type_urban', 0),
        'location_type_suburban': df.get('location_type_suburban', 0)
    }

    feature_df = pd.DataFrame([features])

    # 6. Ensure all model features are present
    for feature in model_features:
        if feature not in feature_df.columns:
            feature_df[feature] = 0

    # 7. Reorder columns to match model's expected input
    feature_df = feature_df[model_features]

    # return feature_df
    return feature_df[model_features], report_delay_days



# ------------------------
# 1. Page Configuration
# ------------------------
st.set_page_config(
    page_title="Submit a new claim - ClaimScope",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📄 Submit a claim for evaluation")

st.markdown("""
**Fill in the details of the insurance claim below to evaluate its fraud risk. All fields are mandatory.**
""")

# ------------------------
# 2. User Input Form
# ------------------------
# split the input data into two columns

with st.form(key='claim_form'):
    col1, col2 = st.columns(2)
    with col1:
        claim_amount = st.number_input(
            "🔢 Claim Amount ($)",
            min_value=0.0,
            step=1.0,
            format="%.2f",
            help="Enter a positive number for the claim amount."
        )

        age = st.number_input(
            "🎂 Age of Policyholder",
            min_value=18,
            max_value=90,
            step=1,
            help="Enter the age of the policyholder (18-90)."
        )

        claim_date = st.date_input(
            "📅 Claim Date",
            min_value=datetime(1900, 1, 1),
            max_value=datetime.today(),
            help="Select the date when the claim occurred."
        )

        report_date = st.date_input(
            "🗓️ Claim Report Date",
            min_value=claim_date,
            max_value=datetime(2090,12,31),
            help="Select the date when the claim was reported."
        )
        
        claim_type = st.selectbox(
            "📂 Claim Type",
            options=["Auto", "Health", "Liability", "Property"],
            help="Select the type of claim."
        )
    with col2:
        adjuster_notes = st.text_area(
            "📝 Adjuster Notes",
            height=206,
            help="Enter any notes made by the adjuster. These notes are analyzed for suspicious terms."
        )



        gender = st.selectbox(
            "👤 Gender",
            options=["Male", "Female", "Unknown"],
            help="Select the gender of the policyholder."
        )

        location_type = st.selectbox(
            "📍 Location Type",
            options=["Urban", "Suburban", "Rural"],
            help="Select the type of location where the claim originated."
        )

    submit_button = st.form_submit_button(label='Evaluate')

# ------------------------
# 3. Handle Form Submission
# ------------------------
if submit_button:
    # Validate inputs (all fields are mandatory)
    if not all([claim_amount, age, claim_date, report_date, adjuster_notes, claim_type, gender, location_type]):
        st.error("⚠️ All fields are mandatory. Please fill in all the details.")
    else:
        # Prepare user input dictionary
        user_input = {
            'claim_amount': claim_amount,
            'age': age,
            'claim_date': claim_date,
            'report_date': report_date,
            'adjuster_notes': adjuster_notes,
            'claim_type': claim_type,
            'gender': gender,
            'location_type': location_type
        }

        # ------------------------
        # 4. Preprocess Input
        # ------------------------
        try:
            model_features = model.feature_names_in_
            feature_vector = preprocess_input(user_input, model_features)
        except Exception as e:
            st.error(f"🚫 Error processing input: {e}")
        else:
            # ------------------------
            # 5. Predict Fraud Risk
            # ------------------------
            try:
                feature_vector, delay_days = preprocess_input(user_input, model_features)
                base_probability = model.predict_proba(feature_vector)[0][1]
                
                # Adjust probability based on delay
                if delay_days <= 30:
                    adjusted_probability = base_probability
                elif delay_days <= 60:
                    adjusted_probability = min(base_probability * 1.5, 1.0)
                elif delay_days <= 90:
                    adjusted_probability = min(base_probability * 2, 1.0)
                else:
                    adjusted_probability = min(base_probability * 4, 1.0)
                    

                probability = adjusted_probability
                
            except Exception as e:
                st.error(f"🚫 Error during prediction: {e}")
            else:
                # ------------------------
                # 6. Display Prediction
                # ------------------------
                if probability == 1:
                    st.error(f"**Fraud Risk Probability: {probability:.2f}**", icon = ":material/search_insights:")
                    st.error(f"🚨 **Based on the data provided, this claim's Fraud Risk Probability score is: {probability:.2f}, indicating it must be reviewed for potential fraud.**")
                
                elif probability >= 0.5:
                    st.warning(f"**Fraud Risk Probability: {probability:.2f}**", icon = ":material/search_insights:")
                    st.warning(f"⚠️ **Based on the data provided, this claim's Fraud Risk Probability score is: {probability:.2f}, indicating it may be fraudulent.**")
                    
                else:
                    st.success(f"**Fraud Risk Probability: {probability:.2f}**", icon = ":material/search_insights:")
                    st.success(f"✅ **Based on the data provided, this claim's Fraud Risk Probability score is: {probability:.2f}, indicating it's probably not fraudulent.**")
                
# ------------------------
# 7. Footer
# ------------------------
st.markdown("---")
st.markdown("""
**ClaimScope** | Developed by [Sami Alyasin](#) | [GitHub Repository](https://github.com/Sami-Alyasin/ClaimScope)
""")