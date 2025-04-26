import streamlit as st
import pandas as pd
import joblib

# 📢 Fix here: Load model without /content
model, feature_columns = joblib.load('salary_model (2).pkl')

# App title
st.title("💼 Salary Predictor 2025")
st.subheader("📈 Predict your salary based on skills, experience, and country")

# Sidebar for user inputs
st.sidebar.header("User Input:")

# User inputs
years_coding = st.sidebar.slider("Years of Coding Experience", 0, 40, 5)

codes_in_python = st.sidebar.checkbox("Do you code in Python?")
codes_in_sql = st.sidebar.checkbox("Do you code in SQL?")
codes_in_java = st.sidebar.checkbox("Do you code in Java?")
codes_in_go = st.sidebar.checkbox("Do you code in Go?")

# Important: country must match how the model was trained
country = st.sidebar.selectbox("Select your Country:", [
    "Country_Nonbinary", 
    "Country_Prefer not to say", 
    "Country_Woman"
])

# Build feature input
input_dict = {
    'Years_Coding': years_coding,
    'Codes_In_Python': int(codes_in_python),
    'Codes_In_SQL': int(codes_in_sql),
    'Codes_In_JAVA': int(codes_in_java),
    'Codes_In_GO': int(codes_in_go),
    'Country_Nonbinary': 0,
    'Country_Prefer not to say': 0,
    'Country_Woman': 0
}

# Set correct country dummy
if country in input_dict:
    input_dict[country] = 1

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Reindex to match training features
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# Predict button
if st.button("💵 Predict Salary"):
    salary_pred = model.predict(input_df)[0]
    st.success(f"💰 Estimated Salary: **${salary_pred:,.2f}**")
