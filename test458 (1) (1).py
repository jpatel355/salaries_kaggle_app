import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def main():
    # --- 1. Model Loading ---
    model_filename = "salary2022_model (2).pkl"
    try:
        with open(model_filename, "rb") as f:
            model = pickle.load(f)
        st.sidebar.success("✅ Model loaded successfully!")
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_filename}' not found. Please ensure it's in the correct location. 📂")
        return
    except Exception as e:
        st.error(f"Error loading model: {e} 🤕")
        return

    # --- 2. Define Constants and Data Structures ---
    st.title("💼 Salary Predictor")
    st.subheader("Accurate salary predictions based on your profile.")

    education_categories = ['HS', 'BS', 'MS', 'PHD']
    education_mapping = {level: i for i, level in enumerate(education_categories)}
    education_options = list(education_mapping.keys())

    all_countries = ["India", "US", "Canada", "Spain", "Germany", "France", "Japan", "UK", "Australia", "Brazil", "Russia", "China", "South Africa", "Netherlands", "Poland", "Italy", "Sweden", "Switzerland", "Turkey", "Mexico", "Argentina", "Colombia", "Indonesia", "Pakistan", "Egypt", "Nigeria", "Iran", "Israel", "Singapore", "Malaysia", "Philippines", "Ireland", "Belgium", "Chile", "Denmark", "Finland", "Norway", "Other"]

    # --- 2b. CRITICAL: Define Expected Features from Training ---
    expected_feature_names = [
        "Codes_In_JAVA", "Codes_In_Python", "Codes_In_SQL", "Codes_In_GO", "Years_Coding", "Education",
        "Country_India", "Country_US", "Country_Spain", "Country_Canada", "Country_Germany",
        "Country_France", "Country_Japan", "Country_UK", "Country_Australia", "Country_Brazil",
        "Country_Russia", "Country_China", "Country_South Africa", "Country_Netherlands",
        "Country_Poland", "Country_Italy", "Country_Sweden", "Country_Switzerland", "Country_Turkey",
        "Country_Mexico", "Country_Argentina", "Country_Colombia", "Country_Indonesia",
        "Country_Pakistan", "Country_Egypt", "Country_Nigeria", "Country_Iran", "Country_Israel",
        "Country_Singapore", "Country_Malaysia", "Country_Philippines", "Country_Ireland",
        "Country_Belgium", "Country_Chile", "Country_Denmark", "Country_Finland"
    ]
    num_expected_features = len(expected_feature_names)

    # --- 3. User Input Widgets ---
    with st.sidebar:
        st.header("Enter Your Details")
        education = st.selectbox("Education Level", education_options)
        years_coding = st.slider("⏳ Years of Coding", 0, 40, 5)
        country = st.selectbox("🌎 Country", all_countries)
        codes_java = st.checkbox("☕ Codes in Java")
        codes_python = st.checkbox("🐍 Codes in Python")
        codes_sql = st.checkbox("📊 Codes in SQL")
        codes_go = st.checkbox("🐹 Codes in Go")

    # --- 4. Prepare Input Data Function ---
    def prepare_input_data(education, years_coding, country, codes_java, codes_python, codes_sql, codes_go, expected_features):
        """
        Prepares the input data for prediction, handling one-hot encoding and ensuring correct feature order.

        Args:
            education (str): User-selected education level.
            years_coding (int): Years of coding experience.
            country (str): User-selected country.
            codes_java (bool): Whether the user codes in Java.
            codes_python (bool): Whether the user codes in Python.
            codes_sql (bool): Whether the user codes in SQL.
            codes_go (bool): Whether the user codes in Go.
