import streamlit as st
import pickle
import pandas as pd

# Load the trained regression model
with open("kaggle2022_model.pkl", "rb") as f:
    model = pickle.load(f)  

# Now continue with the rest of your app
education_mapping = {
    'HS': 0,
    'BS': 1,
    'MS': 2,
    'PHD': 3
}
# Define the education mapping used in training
education_mapping = {
    "I did not complete any formal education": 0,
    "Primary/elementary school": 0,
    "Secondary school": 1,
    "Some college/university study without earning a bachelor’s degree": 1,
    "Bachelor’s degree": 2,
    "Professional degree": 3,
    "Master’s degree": 3,
    "Doctoral degree": 4,
    "No response": 0,
    "Not answered": 0
}

# Available countries based on cleaned data model (excluding reference category 'Canada')
available_countries = ["India", "US", "Spain", "Other", "Canada"]  # Canada is the reference (not one-hot encoded)

# App UI
st.title("💼 Salary Predictor")
st.subheader("📈 Predict your salary based on skills, experience, and education")

# User Inputs
education = st.selectbox("🎓 Education Level", list(education_mapping.keys()))
years_coding = st.slider("💻 Years of Coding Experience", 0, 40, 5)
country = st.selectbox("🌍 Country", available_countries)

codes_java = st.checkbox("Codes in JAVA")
codes_python = st.checkbox("Codes in Python")
codes_sql = st.checkbox("Codes in SQL")
codes_go = st.checkbox("Codes in GO")

# Map education
education_num = education_mapping[education]

# Create feature dictionary
features = {
    "Education": education_num,
    "Years_Coding": years_coding,
    "Codes_In_JAVA": int(codes_java),
    "Codes_In_Python": int(codes_python),
    "Codes_In_SQL": int(codes_sql),
    "Codes_In_GO": int(codes_go),
    "Country_India": 0,
    "Country_US": 0,
    "Country_Spain": 0,
    "Country_Other": 0
}

# Handle one-hot encoding of Country
if country != "Canada":  # Canada is reference (drop_first=True)
    features[f"Country_{country}"] = 1

# Convert to DataFrame for prediction
input_data = pd.DataFrame([features])

# Section header
st.markdown("### 📊 Salary Prediction")
st.write("Click the button below to estimate your salary:")

if st.button("💵 Predict Salary"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Salary: **${prediction:,.2f}**")

# Footer
st.markdown("---")
st.markdown(
    "<small>📘 Built with ❤️ using Streamlit — by Jiya, Rhea, and Michael>",
    unsafe_allow_html=True
)
