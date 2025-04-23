import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("/content/kaggle2022_model.pkl", "rb") as f:
    model = pickle.load(f)

# App title and subtitle
st.title("💼 Salary Predictor (Kaggle Survey)")
st.subheader("📊 Estimate your salary based on skills, experience, and education")

# User inputs
education_mapping = {"HS": 0, "BS": 1, "MS": 2, "PhD": 3}
education = st.selectbox("🎓 Education Level", list(education_mapping.keys()))
education_num = education_mapping[education]

years_coding = st.slider("👨‍💻 Years of Coding Experience", 0, 40, 5)
country = st.selectbox("🌍 Country", ["India", "US", "Spain", "Other"])

codes_python = st.checkbox("🐍 Codes in Python")
codes_sql = st.checkbox("🗄️ Codes in SQL")
codes_java = st.checkbox("☕ Codes in JAVA")
codes_go = st.checkbox("🚀 Codes in GO")

# Feature dictionary setup
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

# Set one-hot encoding for country
if country == "India":
    features["Country_India"] = 1
elif country == "US":
    features["Country_US"] = 1
elif country == "Spain":
    features["Country_Spain"] = 1
else:
    features["Country_Other"] = 1

# Convert to DataFrame
input_df = pd.DataFrame([features])

# Predict button
st.markdown("### 🔮 Predict Salary")
if st.button("💵 Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Salary: **${prediction:,.2f}**")

# Footer
st.markdown("---")
st.markdown("<small>Built with ❤️ using Streamlit</small>", unsafe_allow_html=True)
