"""
preprocess_data.py

Performs basic data cleaning and preprocessing on synthetic_claims.csv.
Outputs a cleaned CSV that can be used for further feature engineering.
"""

import pandas as pd
import numpy as np

def main(input_file="ClaimScope/data/synthetic_claims.csv", output_file="ClaimScope/data/preprocessed_claims.csv"):
    """
    1. Read the raw synthetic data.
    2. Handle basic data cleaning: duplicate removal, dropping rows, etc. based on the results of EDA.
    3. Save the cleaned/preprocessed output as preprocessed_claims.csv.
    """

    # 1. Load Data
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    # 2. Remove Exact Duplicates (if any)
    initial_len = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_len - len(df)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows.")

    # 3. Drop rows that are missing critical fields, e.g. claim_id:
    if "claim_id" in df.columns:
        initial_len = len(df)
        df.dropna(subset=["claim_id"], inplace=True)
        dropped_na = initial_len - len(df)
        if dropped_na > 0:
            print(f"Dropped {dropped_na} rows missing critical 'claim_id' values.")

    # 4. Save the Preprocessed Dataset
    print(f"Saving preprocessed data to {output_file}...")
    df.to_csv(output_file, index=False)

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()