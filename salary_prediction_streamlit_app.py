import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('salary_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set page title
st.title("Data Science Salary Prediction App")

# Create the user input form
st.header("Enter Your Information:")

# Education input
education_level = st.selectbox(
    'What is your highest level of education?',
    [
        "No formal education past high school",
        "Some college/university study without earning a bachelor’s degree",
        "Bachelor’s degree",
        "Master’s degree",
        "Doctoral degree",
        "Professional doctorate",
        "Prefer not to answer"
    ]
)

# Map the education input to numbers (MUST match model)
education_mapping = {
    "No formal education past high school": 0,
    "Some college/university study without earning a bachelor’s degree": 1,
    "Bachelor’s degree": 2,
    "Master’s degree": 3,
    "Doctoral degree": 4,
    "Professional doctorate": 5,
    "Prefer not to answer": -1
}
education_mapped = education_mapping[education_level]

# Country input
country = st.selectbox(
    'Which country are you from?',
    [
        'United States of America',
        'India',
        'United Kingdom of Great Britain and Northern Ireland',
        'Brazil',
        'Canada',
        'Other'
    ]
)

# Predict button
if st.button('Predict Salary'):
    # Create input array for model
    input_data = {
        'Education_Mapped': [education_mapped],
        'Country_Mapped': [country]
    }
   
    import pandas as pd
    input_df = pd.DataFrame(input_data)

    # Make prediction
    predicted_salary = model.predict(input_df)[0]

    # Display result
    st.success(f"Estimated Salary: ${int(predicted_salary):,}")
