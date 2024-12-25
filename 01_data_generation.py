"""
data_generation.py

Generates a synthetic insurance dataset with:
1. Correlated features (e.g., age, location, claim amounts)
2. Noise and outliers
3. Time dependencies (seasonality, incident vs. report date)
4. Missing values

Outputs:
- synthetic_claims.csv
"""

import random
import numpy as np
import pandas as pd
from faker import Faker
from datetime import timedelta, date

# ----------------------------------------
# CONFIGURATIONS
# ----------------------------------------

NUM_POLICYHOLDERS = 5000      
AVG_CLAIMS_PER_CUSTOMER = 2   
FRAUD_RATE = 0.05             
MISSING_RATE = 0.05           
SEED = 42                     

# Seasonality (rough mapping):
#   - Higher property claims in "spring" (storms) and "fall" (hurricanes),
#   - Higher auto claims in "winter" (snow/ice accidents),
#   - We'll define "year" as from -1y to today.

START_DATE = date.today().replace(year=date.today().year - 1)
END_DATE = date.today()

# Initialize Faker and seeds
fake = Faker()
random.seed(SEED)
np.random.seed(SEED)
Faker.seed(SEED)


def generate_policyholder_data(num_customers=5000):
    """
    Generate a list of policyholders with correlated features:
      - age
      - gender
      - location
      - location_risk_score (a hidden factor influencing claim severity)
    Embeds some correlation: older policyholders in certain areas might have fewer claims or different claim types.
    """

    # We’ll define a rough mapping for location_risk_score based on location type (urban, suburban, rural).
    # The higher the risk score, the more likely for high-amount claims.
    LOCATION_TYPES = [
        ("Urban", 0.8),
        ("Suburban", 0.5),
        ("Rural", 0.3)
    ]

    policyholders = []
    for _ in range(num_customers):
        # assign a random location type using set weights
        loc_type, risk_score_base = random.choices(
            LOCATION_TYPES, weights=[0.4, 0.4, 0.2], k=1
        )[0]

        policy_id = fake.unique.uuid4()[:8]
        age = np.random.normal(loc=45, scale=15)
        age = int(max(18, min(age, 90)))         
        gender = random.choice(["M", "F"])
        location_name = fake.city()

        # we'll make the location_risk_score partly correlated with location type, and slightly with age
        # e.g. older individuals might have slightly higher risk if they're still driving
        location_risk_score = risk_score_base + (age - 45) * 0.001
        # ensure risk score stays in [0, 1] range or slightly above
        location_risk_score = max(0.0, min(location_risk_score, 1.5))

        policyholders.append({
            "policy_id": policy_id,
            "age": age,
            "gender": gender,
            "location": f"{location_name} ({loc_type})",
            "location_risk_score": round(location_risk_score, 3)
        })

    return policyholders


def generate_adjuster_notes(claim_type, claim_amount):
    """
    Generate a random adjuster note using 10 predefined templates for each of the 4 claim types.
    """
    
    auto = [
        f"Inspected vehicle for {claim_type}. Damage to the front bumper. Estimated repair cost is ${claim_amount}.",
        f"Claim involves {claim_type}. Noted scratches on driver-side door. Customer reported incident occurred in parking lot.",
        f"Reviewed photos for {claim_type}. Significant damage to rear fender. Claim amount of ${claim_amount} is within policy coverage.",
        f"Incident involves {claim_type}. Confirmed collision with another vehicle at intersection. Claimant provided police report.",
        f"Evaluated claim for {claim_type}. Comprehensive coverage applies due to hailstorm damage.",
        f"Claim reviewed for {claim_type}. Driver reports minor accident during heavy traffic. Repair costs estimated at ${claim_amount}.",
        f"Investigation of {claim_type} claim completed. Uninsured motorist caused the incident. Coverage confirmed.",
        f"Claimant visited our office to discuss {claim_type} claim. Adjuster assigned to inspect vehicle for damage.",
        f"{claim_type} claim submitted for review. Adjuster to confirm coverage for estimated repair cost of ${claim_amount}.",
        f"{claim_type} claim reported by policyholder. Adjuster to follow up on incident details and estimate repair cost.",
    ]
    
    property = [
        f"Claim filed for {claim_type}. Damage to roof caused by severe storm. Estimated repair cost is ${claim_amount}.",
        f"Inspection completed for {claim_type}. Burst pipe caused water damage to kitchen area. Adjuster estimates repairs at ${claim_amount}.",
        f"Reviewed photos for {claim_type}. Fire damage noted in living room. Claim amount of ${claim_amount} approved for processing.",
        f"Claim involves {claim_type}. Policyholder reports theft of personal property. Adjuster assessing value of stolen items.",
        f"Evaluated claim for {claim_type}. Flooding caused damage to basement. Restoration costs estimated at ${claim_amount}.",
        f"Policyholder reported vandalism for {claim_type}. Windows were broken, and repair costs are estimated at ${claim_amount}.",
        f"Inspection for {claim_type} revealed structural damage due to a fallen tree. Adjuster estimated repair costs at ${claim_amount}.",
        f"Fire sprinkler malfunction for {claim_type} caused water damage throughout the office. Estimated restoration costs are ${claim_amount}.",
        f"Policyholder filed a {claim_type} claim for broken HVAC system after a power surge. Adjuster reviewing repair costs at ${claim_amount}.",
        f"Review completed for {claim_type}. Tenant damage to rental property noted. Repair estimates total ${claim_amount}.",
    ]
    
    liability = [
        f"Incident reported for {claim_type}. Policyholder liable for damage to third-party vehicle. Estimated cost ${claim_amount}.",
        f"Reviewing {claim_type} claim. Injury reported at insured property. Adjuster evaluating medical bills and related expenses.",
        f"Adjuster notes for {claim_type}. Policyholder responsible for minor collision. Settlement amount of ${claim_amount} proposed.",
        f"Claim processed for {claim_type}. Adjuster confirmed liability for property damage. Repair costs estimated at ${claim_amount}.",
        f"Investigation of {claim_type} claim completed. Legal review in progress for potential litigation risks.",
        f"Policyholder found liable for damage to neighbor's fence in a {claim_type} claim. Repair costs estimated at ${claim_amount}.",
        f"Review completed for {claim_type}. Slip-and-fall incident at insured location resulted in a settlement proposal of ${claim_amount}.",
        f"Adjuster assessed {claim_type} involving accidental damage to third-party equipment. Estimated compensation is ${claim_amount}.",
        f"Incident at insured business resulted in a {claim_type} claim for foodborne illness. Settlement amount of ${claim_amount} under review.",
        f"Policyholder found liable for injuries caused by a falling object. Medical expenses estimated at ${claim_amount}.",
    ]
    
    health = [
        f"Claim filed for {claim_type}. Medical expenses for surgery total ${claim_amount}. Coverage confirmed under health policy.",
        f"Adjuster reviewing {claim_type}. Claimant submitted bills for emergency room visit. Expenses total ${claim_amount}.",
        f"Claim involves {claim_type}. Ongoing physical therapy sessions for claimant. Estimated costs of ${claim_amount} reviewed.",
        f"Processed {claim_type} claim. Prescription medication costs totaling ${claim_amount} approved under policy.",
        f"Incident involves {claim_type}. Policyholder hospitalized for injury. Medical bills of ${claim_amount} submitted for review.",
        f"Claimant filed a {claim_type} claim for unexpected dental surgery. Adjuster approved coverage for expenses totaling ${claim_amount}.",
        f"Adjuster reviewing {claim_type}. Claimant's expenses for a diagnostic MRI scan total ${claim_amount}.",
        f"Claimant submitted a {claim_type} claim for rehabilitation costs after surgery. Estimated total ${claim_amount} under evaluation.",
        f"Reviewing {claim_type} claim. Medical transport costs due to emergency evacuation are ${claim_amount}.",
        f"Policyholder reported costs for mental health counseling sessions under {claim_type}. Total claim amount is ${claim_amount}.",
    ]
    
    templates = {'auto':auto, 'property':property, 'liability':liability, 'health':health}
    
    return random.choice(templates[claim_type])


def generate_claims(policyholders, avg_claims_per_customer=2):
    """
    Generate claims for each policyholder, embedding:
      - time dependencies (seasonality),
      - correlated claim amount based on location_risk_score,
      - Poisson distribution for # of claims per policyholder,
      - random adjuster notes.
    """

    claims = []

    # Define claim types with approximate seasonal spikes
    # "property" = more in spring/fall, "auto" = more in winter, etc.
    claim_types = ["auto", "property", "liability", "health"]

    for holder in policyholders:
        num_claims = np.random.poisson(avg_claims_per_customer)
        for _ in range(num_claims):
            claim_id = fake.unique.uuid4()[:8]

            # Choose a random date in the past year for incident_date
            incident_date = fake.date_between(start_date=START_DATE, end_date=END_DATE)
            
            # Some basic logic for "seasonality" in choosing claim type
            month = incident_date.month
            if month in [12, 1, 2]:  # winter
                ctype_weights = [0.5, 0.2, 0.2, 0.1]  # auto more likely
            elif month in [3, 4, 5]:  # spring
                ctype_weights = [0.2, 0.5, 0.2, 0.1]  # property more likely
            elif month in [9, 10, 11]:  # fall
                ctype_weights = [0.2, 0.5, 0.2, 0.1]  # property more likely
            else:  # summer
                ctype_weights = [0.3, 0.3, 0.2, 0.2]
            
            claim_type = random.choices(claim_types, weights=ctype_weights, k=1)[0]

            # Report date: typically after the incident, up to 30 days delay
            max_delay_days = 30
            delay_days = random.randint(0, max_delay_days)
            report_date = incident_date + timedelta(days=delay_days)
            if report_date > END_DATE:
                report_date = END_DATE

            # Base claim amount can vary by claim type
            if claim_type == "auto":
                base_amount = np.random.normal(loc=3000, scale=1000)
            elif claim_type == "property":
                base_amount = np.random.normal(loc=8000, scale=3000)
            elif claim_type == "liability":
                base_amount = np.random.normal(loc=5000, scale=1500)
            else:  # health
                base_amount = np.random.normal(loc=6000, scale=2000)

            # location_risk_score modifies the base amount
            # e.g., a higher risk score => on average, 20% higher claims
            location_risk_score = holder["location_risk_score"]
            claim_amount = round(base_amount * (1 + 0.2 * location_risk_score), 2)
            claim_amount = max(200, claim_amount)  # clamp to a minimum

            # generate adjuster notes using the function we created
            adjuster_notes = generate_adjuster_notes(claim_type, claim_amount)

            claims.append({
                "policy_id": holder["policy_id"],
                "claim_id": claim_id,
                "incident_date": incident_date,
                "report_date": report_date,
                "claim_type": claim_type,
                "claim_amount": round(claim_amount, 2),
                "adjuster_notes": adjuster_notes
            })

    return claims


def introduce_fraud(claims, fraud_rate=0.05):
    """
    Label a subset of claims as fraud, injecting patterns:
      - Exaggerated claim amounts,
      - Suspicious keywords in adjuster notes,
      - Possibly contradictory or unrealistic dates.
    """

    total_claims = len(claims)
    fraud_count = int(total_claims * fraud_rate)
    fraud_indices = random.sample(range(total_claims), fraud_count)

    for idx in range(total_claims):
        if idx in fraud_indices:
            claims[idx]["is_fraud"] = 1

            # Exaggerate claim amount
            multiplier = random.uniform(1.5, 3.0)
            claims[idx]["claim_amount"] = round(claims[idx]["claim_amount"] * multiplier, 2)

            # Insert suspicious keywords
            suspicious_keywords = [
                "inconsistent statements",
                "multiple prior incidents",
                "fabricated evidence",
                "urgent payout demanded"
            ]
            claims[idx]["adjuster_notes"] += " " + random.choice(suspicious_keywords)

            # Occasionally flip dates to be suspicious
            if random.random() < 0.1:
                # Make report_date before incident_date
                incident = claims[idx]["incident_date"]
                report = claims[idx]["report_date"]
                claims[idx]["incident_date"] = report
                claims[idx]["report_date"] = incident
        else:
            claims[idx]["is_fraud"] = 0

    return claims


def introduce_noise_and_outliers(claims, outlier_fraction=0.02):
    """
    Introduce random noise and outliers in claim_amount and dates.
    - A small fraction of claims might be extremely high or negative (data entry errors).
    - Random date mistakes.
    """

    total_claims = len(claims)
    outlier_count = int(total_claims * outlier_fraction)
    outlier_indices = random.sample(range(total_claims), outlier_count)

    for idx in range(total_claims):
        # Random small noise in claim_amount
        noise_factor = np.random.normal(loc=1, scale=0.05)  # ±5% typical
        claims[idx]["claim_amount"] = round(claims[idx]["claim_amount"] * noise_factor, 2)

        # For outliers
        if idx in outlier_indices:
            # Randomly decide if it's an extremely large outlier or negative
            if random.random() < 0.5:
                # Large outlier
                claims[idx]["claim_amount"] = round(claims[idx]["claim_amount"] * random.uniform(5, 10), 2)
            else:
                # Negative or zero due to data error
                claims[idx]["claim_amount"] = round(random.uniform(-5000, 0), 2)

            # Possibly scramble dates
            if random.random() < 0.3:
                claims[idx]["incident_date"] = fake.date_between(start_date=END_DATE, end_date=END_DATE)
                claims[idx]["report_date"] = fake.date_between(start_date=END_DATE, end_date=END_DATE)

    return claims


def introduce_missing_values(df, missing_rate=0.05):
    """
    Randomly replace a fraction of the dataframe cells with NaN
    to simulate real-world data. Only applied to the following fields:
    - policy_id
    - claim_id
    - incident_date
    - report_date
    - age
    - gender
    - location
    - location_risk_score
    """

    # List of columns to apply the function to
    applicable_columns = [
            'policy_id', 'claim_id', 'incident_date', 
            'report_date', 'age', 'gender', 
            'location', 'location_risk_score'
        ]
    
    # Flatten the dataframe into an array
    arr = df.to_numpy().astype(object)
    total_elements = sum([df[col].size for col in applicable_columns if col in df.columns])

    # Number of elements to be replaced with NaN
    num_missing = int(total_elements * missing_rate)

    # Get indices of applicable columns
    applicable_col_indices = [df.columns.get_loc(col) for col in applicable_columns if col in df.columns]

    # Select random positions for missing values only in applicable columns
    missing_positions = []
    while len(missing_positions) < num_missing:
        row_idx = random.randint(0, df.shape[0] - 1)
        col_idx = random.choice(applicable_col_indices)
        missing_positions.append((row_idx, col_idx))

    for row_idx, col_idx in missing_positions:
        arr[row_idx, col_idx] = np.nan

    # Reconstruct the DataFrame
    df_missing = pd.DataFrame(arr, columns=df.columns)
    return df_missing


def main():
    print("Generating policyholder data...")
    policyholders = generate_policyholder_data(NUM_POLICYHOLDERS)

    print("Generating claims data with time dependencies, correlation, and seasonality...")
    claims = generate_claims(policyholders, AVG_CLAIMS_PER_CUSTOMER)

    print("Introducing fraud with exaggerated amounts and suspicious notes...")
    claims = introduce_fraud(claims, FRAUD_RATE)

    print("Introducing random noise, outliers, and date anomalies...")
    claims = introduce_noise_and_outliers(claims, outlier_fraction=0.02)

    print("Converting to DataFrame...")
    df = pd.DataFrame(claims)

    # Combine with policyholder data
    # Merge on policy_id to incorporate age, gender, location, location_risk_score
    ph_df = pd.DataFrame(policyholders)
    merged_df = pd.merge(df, ph_df, on="policy_id", how="left")

    print("Introducing missing values...")
    merged_df = introduce_missing_values(merged_df, MISSING_RATE)

    print("Saving synthetic_claims.csv...")
    
    merged_df.to_csv("~/projects/ClaimScope/data/synthetic_claims.csv", index=False)
    print("Data generation complete. File: synthetic_claims.csv")


if __name__ == "__main__":
    main()