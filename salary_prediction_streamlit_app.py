import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open('gradient_boosting_model_with_experience.pkl', 'rb') as f:
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
        "I prefer not to answer"
    ]
)

# Map education input to numbers
education_mapping = {
    "No formal education past high school": 0,
    "Some college/university study without earning a bachelor’s degree": 1,
    "Bachelor’s degree": 2,
    "Master’s degree": 3,
    "Doctoral degree": 4,
    "Professional doctorate": 5,
    "I prefer not to answer": -1
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

# Machine Learning Experience input
experience_years = st.selectbox(
    'For how many years have you used machine learning methods?',
    [
        'I do not use machine learning methods',
        'Under 1 year',
        '1-2 years',
        '2-3 years',
        '3-4 years',
        '4-5 years',
        '5-10 years',
        '10-20 years',
        '20 or more years'
    ]
)

# Map experience input to numbers
experience_mapping = {
    'I do not use machine learning methods': 0,
    'Under 1 year': 0.5,
    '1-2 years': 1.5,
    '2-3 years': 2.5,
    '3-4 years': 3.5,
    '4-5 years': 4.5,
    '5-10 years': 7.5,
    '10-20 years': 15,
    '20 or more years': 25
}
experience_mapped = experience_mapping[experience_years]

# Predict button
if st.button('Predict Salary'):
    # Create input array for model
    input_data = {
        'Education_Mapped': [education_mapped],
        'Country_Mapped': [country],
        'Experience_Mapped': [experience_mapped]
    }
   
    input_df = pd.DataFrame(input_data)

    # Make prediction
    predicted_salary = model.predict(input_df)[0]

    # Display result
    st.success(f"Estimated Salary: ${int(predicted_salary):,}")
