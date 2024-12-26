
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import os

# ------------------------
# 1. Page Configuration
# ------------------------
st.set_page_config(layout="wide",initial_sidebar_state="expanded",page_icon="")

# ------------------------
# 2. Project Title and Description
# ------------------------
st.markdown("<h1 style='text-align: center;'> ClaimScope - Insurance Fraud Detection 🔍</h1>", unsafe_allow_html=True)    
st.divider()
# insert two columns with some space in between
col1, col2, col3 = st.columns([9, 1, 9])
with col1:
    st.header("📝 About This Project", divider=True)
    st.markdown("""
    **ClaimScope** leverages machine learning to identify potentially fraudulent insurance claims, helping insurers mitigate losses and streamline their operations. 
    By analyzing various features of insurance claims, ClaimScope predicts the likelihood of fraud, enabling proactive decision-making.
    """)

    # ------------------------
    # 3. Introduction to Insurance Fraud Detection
    # ------------------------
    st.header("📊 Understanding Insurance Fraud", divider=True)

    st.markdown("""
    Insurance fraud is a significant issue that costs the industry billions annually. 
    Fraudulent claims not only inflate insurance premiums but also erode trust between insurers and policyholders. 
    Detecting and preventing such fraudulent activities is crucial for maintaining the integrity and sustainability of insurance services.
    """)

# ------------------------
# 4. How ClaimScope Works
# ------------------------
with col3:
    st.header("🔍 How ClaimScope Works", divider=True)

    st.markdown("""
    ClaimScope employs a **Random Forest Classifier** trained on historical insurance claim data to predict the likelihood of a claim being fraudulent. 
    The model analyzes key features of each claim, such as:

    - **Claim Amount**
    - **Age of the Policyholder**
    - **Location Risk Score**
    - **Report Delay Days**
    - **Claim Type (e.g., Auto, Home)**
    - **Gender**
    - **Location Type (e.g., Urban, Rural)**
    - **Suspicious Note Flags**

    These features help the model discern patterns and anomalies indicative of fraudulent behavior.
    """)

# ------------------------
# 5. Key Feature Importances
# ------------------------
st.header("📈 Key Feature Importances", divider=True)

st.markdown("""
The chart below showcases the most influential features that drive the model's predictions. Understanding these features helps stakeholders grasp the factors contributing to potential fraud detection.
""")

# load the feature importances and plot the data
feature_importances = pd.read_csv("https://raw.githubusercontent.com/Sami-Alyasin/ClaimScope/refs/heads/main/data/feature_importances.csv")
# Sort the DataFrame by the importance values in descending order
sorted_importances = feature_importances.sort_values(by="Importance", ascending=False)

# plot using altair
chart = alt.Chart(sorted_importances).mark_bar().encode(
    x='Importance',
    y=alt.Y('Feature', sort='-x', title='Feature Names'),
    color=alt.Color('Feature', legend=None)
).configure_axis(labelLimit=500 # Increase the limit for axis labels to avoid truncation
                    ).properties(
height=600  # Adjust the height if needed
)

st.altair_chart(chart, use_container_width=True)
    
# ------------------------
# 6. How to Use the App
# ------------------------
st.header("🚀 How to Use This App", divider=True)

st.markdown("""
**Get Started by navigating to the pages available from the sidebar!** 

Currently, you can do the following in this app:
- **🔍 Evaluate Claim:** Predict the likelihood of fraud for a given insurance claim based on user input.
- **🧑‍💻Code Walkthrough:** Explore the codebase and the steps involved in building ClaimScope.

""")

# ------------------------
# 7. Footer
# ------------------------
st.markdown("---")
st.markdown("""
**ClaimScope** | Developed by [Sami Alyasin](#) | [GitHub Repository](https://github.com/Sami-Alyasin/ClaimScope)
""")