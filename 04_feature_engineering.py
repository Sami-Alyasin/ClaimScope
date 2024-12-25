"""
04_feature_engineering.py

Creates new features, handles missing data, and outputs a cleaned/processed CSV with the new features.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def main():
    # 1. Load the raw dataset
    df = pd.read_csv("ClaimScope/data/preprocessed_claims.csv")

    # 2. Create New Features

    # Delay between incident and report
    # Convert incident_date and report_date to datetime
    df["incident_date"] = pd.to_datetime(df["incident_date"])
    df["report_date"] = pd.to_datetime(df["report_date"])

    # If there's any case where incident_date > report_date, this might be from fraudulent flips or data anomalies.
    df["report_delay_days"] = (df["report_date"] - df["incident_date"]).dt.days
    # replace the nulls with 0 and then convert the column to an intiger
    df["report_delay_days"] = df["report_delay_days"].fillna(0).astype(int)

    # Extract suspicious keywords from adjuster notes
    # We'll create a feature if the note contains certain suspicious terms
    # Convert adjuster_notes to a string first in case of nulls
    df["adjuster_notes"] = df["adjuster_notes"].astype(str)
    suspicious_terms = ["inconsistent statements", "fabricated evidence", "urgent payout demanded"]
    df["suspicious_note_flag"] = df["adjuster_notes"].apply(
        lambda x: 1 if any(term in x.lower() for term in suspicious_terms) else 0
    )

    # Extract the location type from the location (urban, suburban, rural)
    # since we have nulls, we'll fill them with "unknown", then we can apply the function to extract the location type
    df["location"] = df["location"].fillna("unknown")
    df["location_type"] = df["location"].apply(
        lambda x: "urban" if "urban" in x.lower() else "suburban" if "suburban" in x.lower() else "rural"
    )

    # 3. Handle Missing Values

    # create a list of columns with missing values
    cols = df.isnull().sum().index[df.isnull().sum() > 0]

    # We'll separate numeric and categorical columns for different imputation strategies and filter only on columns with missing values
    numeric_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[cols].select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Fill numeric nulls with the median
    numeric_imputer = SimpleImputer(strategy="median")
    df['age'] = numeric_imputer.fit_transform(df[numeric_cols])
    
    # For location_risk_score, if the location is not null:
    #   we can fill the missing values with the mean of the location_risk_score
    df["location_risk_score"] = df["location_risk_score"].fillna(df.groupby("location")["location_risk_score"].transform("mean"))

    #   otherwise we can fill the missing values with the mean of the entire column 
    df["location_risk_score"] = df["location_risk_score"].fillna(df["location_risk_score"].mean())

    # Fill categorical nulls with "Unknown"
    # handle dates first. If one is missing, fill with the other. We already dropped the rows with both missing in the preprocessing step.
    df["incident_date"] = df["incident_date"].fillna(df["report_date"])
    df["report_date"] = df["report_date"].fillna(df["incident_date"])

    # fill the remaining categorical columns with "unknown"
    for col in categorical_cols:
        df[col] = df[col].fillna("unknown")

    # 4. Save Processed Data
    df.to_csv("ClaimScope/data/processed_claims.csv", index=False)
    print("Feature engineering complete. File saved: processed_claims.csv")

if __name__ == "__main__":
    main()
