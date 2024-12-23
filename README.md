# ClaimScope: Insurance Fraud Detection

**End-to-End Data Science Project for Identifying Fraudulent Claims Using Synthetic Insurance Data**

## Overview

`ClaimScope` is a comprehensive, end-to-end project that demonstrates how to detect fraudulent insurance claims. By simulating an insurance dataset, it covers the entire data science lifecycle—ranging from synthetic data generation to building, training, and deploying machine learning models. 

Refer to `status.md` to see the current progress


## Table of Contents

- [Motivation](#motivation)  
- [Features](#features)  
- [Project Architecture](#project-architecture)  
- [Installation & Setup](#installation--setup)  
- [Data Generation](#data-generation)  
- [Usage](#usage)  
- [Model Training & Evaluation](#model-training--evaluation)  
- [Deployment](#deployment)  
- [Project Structure](#project-structure)  
- [Future Improvements](#future-improvements)  
- [License](#license)

---

## Motivation

1. **Realistic Demonstration**: Insurance claims data is usually confidential. This project uses synthetic data to simulate real-world complexity while protecting privacy.  
2. **End-to-End Pipeline**: Showcases skills in data engineering, ML modeling, and model deployment.  
3. **Industry Relevance**: Fraud detection is a critical problem in insurance, with high potential ROI for data-driven solutions.

---

## Features

- **Synthetic Data Generation**: Create realistic policyholder and claims data with embedded fraud patterns.  
- **Fraud Detection Models**: Implement and compare multiple classification models (Logistic Regression, Random Forest, XGBoost, etc.).  
- **Feature Engineering & Text Analysis**: Analyze structured and unstructured fields (e.g., “adjuster notes”) for potential fraud indicators.  
- **Model Evaluation & Explainability**: Use metrics like precision, recall, AUC, and confusion matrices. Integrate explainability tools (like SHAP).  
- **Deployment**: Demonstrate a simple web application (e.g., Streamlit or Flask) to visualize claims data and flag high-risk cases.

---

## Project Architecture

```
   ┌────────────────────────┐
   │  Synthetic Data (CSV)  │
   └────────────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Data Preprocessing & Cleaning  │
└─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│   Feature Engineering           │
│  (Structured + Text Features)   │
└─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Model Training & Evaluation    │
│ (Classifiers + Explainability)  │
└─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│   Deployment (Web App / API)    │
│   + Visualization Dashboard     │
└─────────────────────────────────┘
```

---

## Installation & Setup

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/Sami-Alyasin/ClaimScope.git
   cd ClaimScope
   ```
2. **Create & Activate a Virtual Environment (Optional)**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Generation

Because real insurance claim data is sensitive, we generate synthetic data using the [Faker](https://faker.readthedocs.io/en/master/) library (or any preferred tool). 

- **File**: `data_generation.py`  
- **Main Steps**:  
  1. Generate policyholder data (IDs, demographics, location).  
  2. Create claims records (dates, amounts, types, adjuster notes).  
  3. Inject fraud cases by altering patterns (high claim amounts, suspicious notes, etc.).  

To run data generation:
```bash
python data_generation.py
```
This produces a CSV file (e.g., `synthetic_claims.csv`) that will be used for modeling.

---

## Usage

1. **Data Preprocessing**  
   Run the preprocessing script to clean and transform the raw CSV:
   ```bash
   python preprocess_data.py --input synthetic_claims.csv --output preprocessed_claims.csv
   ```

2. **Model Training**  
   Train your classification models (e.g., Logistic Regression, Random Forest, XGBoost):
   ```bash
   python train_model.py --data preprocessed_claims.csv --model_type random_forest
   ```

3. **Evaluation**  
   View evaluation metrics (accuracy, precision, recall, AUC, etc.) in your console or generated reports in the `reports/` directory.

---

## Model Training & Evaluation

- **Algorithms**: Logistic Regression, Decision Trees, Random Forest, XGBoost (or others you choose).  
- **Metrics**:  
  - *Confusion Matrix:* to observe true vs. false positives.  
  - *Precision & Recall:* crucial for fraud detection (catch fraudulent claims without over-flagging legitimate ones).  
  - *ROC AUC:* overall performance indicator.  
- **Explainability**: Tools like [SHAP](https://github.com/slundberg/shap) to highlight key features contributing to fraud predictions.

---

## Deployment

A simple **Streamlit** app can be used to demonstrate real-time fraud checks:

1. **Local Deployment**  
   ```bash
   streamlit run app.py
   ```
   - Navigate to `http://localhost:8501` (default) to interact with the dashboard.  

2. **Cloud Deployment**  
   - Deploy the app to Streamlit Community Cloud
   - If you haven't done this before, start [here](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)

**Dashboard Features**:  
- Upload new claims data and see the model’s fraud risk scores.  
- Visualize important features, trending patterns, and suspicious keywords from text fields.

---

## Project Structure

```
ClaimScope/
│
├── data/                    # Synthetic CSV files
├── reports/                 # Evaluation reports (metrics, plots)
├── notebooks/               # Jupyter notebooks for EDA & experimentation
├── pages/                   # Pages of streamlit app
├── models/                  # Saved model artifacts
├── app.py                   # Streamlit app for deployment
├── data_generation.py       # Script for creating synthetic dataset
├── preprocess_data.py       # Cleans and structures raw CSV
├── train_model.py           # Training & validation script
├── requirements.txt         # Python dependencies
├── status.md                # Status of the project
└── README.md                # Project documentation
```

---

## Future Improvements

1. **Advanced NLP**: Use transformer-based models (e.g., BERT) for more nuanced insights from adjuster notes.  
2. **MLOps Integration**: Include CI/CD pipelines, Dockerization, and automated model retraining with incremental data.  
3. **Real-Time Scoring**: Stream data into a microservice architecture for immediate fraud risk assessment.

---

**Questions or Feedback?**  
Open an issue or submit a pull request on GitHub. Enjoy exploring `ClaimScope`!