# edusalary_streamlit.py

import streamlit as st
import joblib  # <-- NEW: using joblib
import numpy as np

# Load the model
model = joblib.load('kaggle_edu_model_2025_reexported.joblib')

def predict_salary(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

def main():
    st.title("Educational Salary Prediction App 🎓💰")
    st.write("Fill in your details below to predict your expected salary:")

    years_experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.5)
    education_level = st.selectbox(
        "Highest Education Level Achieved",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Bachelor's", 2: "Master's", 3: "PhD"}[x]
    )
    age = st.number_input("Age", min_value=18, max_value=80, step=1)

    if st.button("Predict Salary"):
        features = [years_experience, education_level, age]
        salary = predict_salary(features)
        st.success(f"🎯 Predicted Salary: ${salary:,.2f}")

if __name__ == "__main__":
    main()
