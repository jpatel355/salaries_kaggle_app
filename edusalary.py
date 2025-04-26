import streamlit as st
import pickle
import pandas as pd

# Load the trained regression model
# Adjust the path to where you saved your pickle file.  Use a raw string.
model_path = 'best_salary_prediction_model.pkl'  # Or the path you used
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Define the education mapping (consistent with your training)
education_mapping = {'HS': 0, 'BS': 1, 'MS': 2, 'PhD': 3, 'NR': -1}  # Added 'NR'

# App title
st.title("💼 Salary Predictor")
st.subheader("💰 Predict salary based on your data")

# User input widgets
education = st.selectbox("Education Level", list(education_mapping.keys()))
years_coding = st.slider("Years of Coding Experience", 0, 40, 5)
country = st.selectbox("Country", ["India", "Nigeria", "Other", "United States of America", "Brazil"])

# Map the selected education level to its numeric value
education_num = education_mapping[education]

# Build the feature dictionary for prediction
input_data = pd.DataFrame({
        'YearsCoding': [years_coding],
        'Education': [education_num],
        'Country_Brazil': [1 if country == 'Brazil' else 0],
        'Country_India': [1 if country == 'India' else 0],
        'Country_Nigeria': [1 if country == 'Nigeria' else 0],
        'Country_Other': [1 if country == 'Other' else 0],
        'Country_United States of America': [1 if country == 'United States of America' else 0],
    })


# Section header
st.markdown("### 📊 Salary Prediction")

# Instructions + Predict button
st.write("Click the button below to estimate your salary.")

if st.button("💵 Predict Salary"):
    try:
        prediction = model.predict(input_data)[0]  # Get the scalar value
        st.success(f"💰 Estimated Salary: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please check your input data and model compatibility.")
