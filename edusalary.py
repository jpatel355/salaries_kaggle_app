import streamlit as st
import pickle
import numpy as np

# Load the model
with open('kaggle_edu_model_2025 (1).pkl', 'rb') as file:
    model = pickle.load(file)

def predict_salary(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

def main():
    st.title("Salary Prediction App")

    st.write("Provide the following inputs:")

    years_experience = st.number_input("Years of Experience", min_value=0.0, step=0.1)
    education_level = st.selectbox("Education Level", [1, 2, 3], format_func=lambda x: {1:"Bachelor", 2:"Master", 3:"PhD"}[x])
    age = st.number_input("Age", min_value=0, step=1)

    if st.button("Predict Salary"):
        features = [years_experience, education_level, age]
        salary = predict_salary(features)
        st.success(f"Predicted Salary: ${salary:,.2f}")

if __name__ == "__main__":
    main()
