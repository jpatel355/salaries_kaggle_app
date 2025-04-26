import streamlit as st
import pandas as pd
import joblib

# Load your model + the list of feature columns
model, feature_columns = joblib.load('salary_model (2).pkl')

st.title("💼 Salary Predictor 2025")
st.subheader("📈 Predict your salary based on skills, experience, and country")

st.sidebar.header("User Input:")

# 1. Slider & checkboxes as before
years_coding    = st.sidebar.slider("Years of Coding Experience", 0, 40, 5)
codes_in_python = st.sidebar.checkbox("Do you code in Python?")
codes_in_sql    = st.sidebar.checkbox("Do you code in SQL?")
codes_in_java   = st.sidebar.checkbox("Do you code in Java?")
codes_in_go     = st.sidebar.checkbox("Do you code in Go?")

# 2. Dynamically extract all "Country_*" columns
country_cols   = [c for c in feature_columns if c.startswith("Country_")]
country_labels = [c.replace("Country_", "") for c in country_cols]

# 3. Let user pick one of those real labels
country = st.sidebar.selectbox("Select your Country:", country_labels)

# 4. Build your input dict, zero-fill all country dummies
input_dict = {
    'Years_Coding':   years_coding,
    'Codes_In_Python': int(codes_in_python),
    'Codes_In_SQL':    int(codes_in_sql),
    'Codes_In_JAVA':   int(codes_in_java),
    'Codes_In_GO':     int(codes_in_go),
}
for col in country_cols:
    input_dict[col] = 0

# 5. Turn on exactly the chosen country dummy
input_dict[f"Country_{country}"] = 1

# 6. DataFrame → reindex → predict
input_df = pd.DataFrame([input_dict]).reindex(columns=feature_columns, fill_value=0)

if st.button("💵 Predict Salary"):
    salary_pred = model.predict(input_df)[0]
    st.success(f"💰 Estimated Salary: **${salary_pred:,.2f}**")
