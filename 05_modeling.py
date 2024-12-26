"""
05_modeling.py

Trains classification models for fraud detection on processed_claims.csv,
evaluates performance, and saves model artifacts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

def main():
    # 1. Load Processed Data
    df = pd.read_csv("/Users/sami/projects/ClaimScope/data/processed_claims.csv")

    # 2. Define Features and Target

    X = df.drop(columns=["is_fraud", "policy_id", "claim_id", "adjuster_notes","incident_date", "report_date", "location"])
    y = df["is_fraud"].astype(int)

    # 3. Train/Test Split
    # We're making sure to use stratify here since the faudulent data is only present in abobut 5% of the dataset
    # this way we ensure that those records are represented in the same proportion in both the training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y 
    )

    # 4. Scale Continious Numeric Features (Necessary for Logistic Regression)
    numeric_cols = ["claim_amount","age","location_risk_score","report_delay_days"]
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # 5. One-Hot Encode Categorical Features
    X_train = pd.get_dummies(X_train, columns=["claim_type","gender","location_type"])
    X_test = pd.get_dummies(X_test, columns=["claim_type","gender","location_type"])

    # 6. Train Two Example Models: Logistic Regression & Random Forest

    # 6.1. Logistic Regression
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train, y_train)

    # 6.2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # 7. Evaluate Models
    print("=== Logistic Regression Evaluation ===")
    evaluate_model(logreg, X_test, y_test)

    print("\n=== Random Forest Evaluation ===")
    evaluate_model(rf, X_test, y_test)

    # 8. Save the Best Model (Assume Random Forest for demonstration)
    with open("ClaimScope/models/fraud_detector_model.pkl", "wb") as f:
        pickle.dump(rf, f)

    # Save the scaler as well
    with open("ClaimScope/models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\nModel training complete. Best model saved as 'fraud_detector_model.pkl'.")
    
    #9. Generate feature importance visual
    feature_columns = X_train.columns.to_list()

    feature_importances = rf.feature_importances_
    feature_importances

    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # plot feature importances
    plt.figure(figsize=(20, 6))
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_importance_df, palette='viridis', dodge=False, legend=False)
    plt.title('Feature Importance from Random Forest Regressor', fontsize=16, fontweight='bold')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.gca().set_facecolor('#f0f0f0')  # Set background color
    plt.grid(True, linestyle='--', alpha=0.7)   
    plt.savefig("ClaimScope/visuals/feature_importance.png")
    
    
def evaluate_model(model, X_test, y_test):
    """
    Prints confusion matrix, classification report, and ROC-AUC for the given model.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    if y_proba is not None:
        roc_score = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC Score: {roc_score:.4f}")
    else:
        print("ROC-AUC Score: N/A (model has no predict_proba)")

if __name__ == "__main__":
    main()