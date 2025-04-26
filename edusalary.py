import streamlit as st
import pickle
import pandas as pd

# Load the trained regression model
with open("kaggle_edu_model_2022.pkl", "rb") as f:
    model = pickle.load(f)

# Define the education mapping (example: adapt if different in your model)
education_mapping = {'HS': 0, 'BS': 1, 'MS': 2, 'PHD': 3}

# App title
st.title("🎓 Education Salary Predictor 2025")
st.subheader("📈 Estimate your salary based on education, coding skills, and experience")

# User input widgets
education = st.selectbox("Education Level", list(education_mapping.keys()))
years_coding = st.slider("Years of Coding Experience", 0, 40, 5)
country = st.selectbox("Country", ["India", "US", "Canada", "Spain", "Other"])
codes_java = st.checkbox("Codes in JAVA")
codes_python = st.checkbox("Codes in Python")
codes_sql = st.checkbox("Codes in SQL")
codes_go = st.checkbox("Codes in GO")

# Map the selected education level to its numeric value
education_num = education_mapping[education]

# Build the feature dictionary for prediction
features = {
    "Education": education_num,
    "Years_Coding": years_coding,
    "Codes_In_JAVA": int(codes_java),
    "Codes_In_Python": int(codes_python),
    "Codes_In_SQL": int(codes_sql),
    "Codes_In_GO": int(codes_go),
    "Country_India": 0,
    "Country_Other": 0,
    "Country_Spain": 0,
    "Country_US": 0,
}

# Set country dummy variables
if country != "Canada":
    if country == "India":
        features["Country_India"] = 1
    elif country == "US":
        features["Country_US"] = 1
    elif country == "Spain":
        features["Country_Spain"] = 1
    elif country == "Other":
        features["Country_Other"] = 1

# Create DataFrame
input_data = pd.DataFrame([features])

# Section header
st.markdown("### 📊 Salary Prediction")

# Predict button
st.write("Click below to estimate your salary:")
if st.button("💵 Predict Salary"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Salary: **${prediction:,.2f}**")

# Footer
st.markdown("---")
st.markdown(
    "<small>📘 Built with ❤️ using Streamlit — 2025 Edition</small>",
    unsafe_allow_html=True
)
