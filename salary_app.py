import streamlit as st
import pandas as pd
import pickle

# Load the model
with open("kaggle2022_model(2).pkl", "rb") as f:
    model = pickle.load(f)

# Display expected model features
expected_features = list(model.feature_names_in_)
st.sidebar.markdown("### 🔍 Model expects these features:")
st.sidebar.write(expected_features)

# Title
st.title("💼 Salary Prediction App (Kaggle Survey 2022)")

# User Inputs
age = st.selectbox("Select Age Range", ["18-21", "22-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-69", "70+"])
country = st.selectbox("Country", ["India", "US", "Spain", "Other"])
student_status = st.selectbox("Are you a student?", ["Yes", "No"])
codes_python = st.checkbox("I code in Python")
codes_sql = st.checkbox("I code in SQL")
codes_java = st.checkbox("I code in Java")
codes_go = st.checkbox("I code in Go")
years_coding = st.slider("Years of Coding Experience", 0, 50, 2)

# Prepare input features
input_data = {
    "Age": age,
    "Country_India": 1 if country == "India" else 0,
    "Country_Spain": 1 if country == "Spain" else 0,
    "Country_US": 1 if country == "US" else 0,
    "Country_Other": 1 if country == "Other" else 0,
    "Student_Status": 1 if student_status == "Yes" else 0,
    "Codes_In_Python": int(codes_python),
    "Codes_In_SQL": int(codes_sql),
    "Codes_In_JAVA": int(codes_java),
    "Codes_In_GO": int(codes_go),
    "Years_Coding": years_coding,
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Debugging: Show input
st.write("📊 Input DataFrame")
st.write(input_df)

# Check feature match
input_columns = input_df.columns.tolist()
if set(expected_features) != set(input_columns):
    st.error("❌ Input features do NOT match the model's expected features.")
    st.write("Expected:", expected_features)
    st.write("Received:", input_columns)
else:
    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Salary: ${int(prediction):,}")
