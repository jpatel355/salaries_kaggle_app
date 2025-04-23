import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model
try:
    with open("kaggle2022_model (2).pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

# 1.  IMPORTANT:  Replace this with the ACTUAL expected features from the training data!
#     This is just an example.
expected_features = [
    "Age",
    "Country_India", "Country_Spain", "Country_US", "Country_Other", "Country_France",  # Added France
    "Country_Germany", "Country_Japan",  # Added Germany, Japan
    "Student_Status",
    "Codes_In_Python", "Codes_In_SQL", "Codes_In_JAVA", "Codes_In_GO",
    "Years_Coding",
    #  Example if Age was one-hot encoded in training:
    #"Age_18-21", "Age_22-24", "Age_25-29", "Age_30-34", "Age_35-39", "Age_40-44", "Age_45-49", "Age_50-54", "Age_55-59", "Age_60-69", "Age_70+"
]

# Create Streamlit app
def main():
    st.title("Salary Prediction App (Kaggle Survey 2022)")

    # Sidebar
    st.sidebar.markdown("### Model Information")
    st.sidebar.markdown("This app predicts salary...")
    st.sidebar.markdown("### Expected Features:")
    st.sidebar.write(expected_features)

    # User Inputs
    age = st.selectbox("Age Range", ["18-21", "22-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-69", "70+"])
    country = st.selectbox("Country", ["India", "US", "Spain", "Other", "France", "Germany", "Japan"]) # Added countries
    student_status = st.selectbox("Student?", ["Yes", "No"])
    codes_python = st.checkbox("Python")
    codes_sql = st.checkbox("SQL")
    codes_java = st.checkbox("Java")
    codes_go = st.checkbox("Go")
    years_coding = st.slider("Coding Experience", 0, 50, 2)

    # Age conversion
    age_mapping = {
        "18-21": 18, "22-24": 22, "25-29": 25, "30-34": 30, "35-39": 35,
        "40-44": 40, "45-49": 45, "50-54": 50, "55-59": 55, "60-69": 60, "70+": 70
    }
    age_num = age_mapping[age]

    # 2.  IMPORTANT:  Modify this to create ALL the columns the model expects.
    #     This is just an example.  You MUST get the country list from the
    #     training data.
    input_data = {
        "Age": age_num,
        "Country_India": 1 if country == "India" else 0,
        "Country_Spain": 1 if country == "Spain" else 0,
        "Country_US": 1 if country == "US" else 0,
        "Country_Other": 1 if country == "Other" else 0,
        "Country_France": 1 if country == "France" else 0, # Added France
        "Country_Germany": 1 if country == "Germany" else 0, # Added Germany
        "Country_Japan": 1 if country == "Japan" else 0,     # Added Japan
        "Student_Status": 1 if student_status == "Yes" else 0,
        "Codes_In_Python": int(codes_python),
        "Codes_In_SQL": int(codes_sql),
        "Codes_In_JAVA": int(codes_java),
        "Codes_In_GO": int(codes_go),
        "Years_Coding": years_coding,
        # If Age was one-hot encoded:
        #"Age_18-21": 1 if age == "18-21" else 0,
        #"Age_22-24": 1 if age == "22-24" else 0,
        # ... and so on for all age ranges
    }

    input_df = pd.DataFrame([input_data])

    st.write("Input Data")
    st.dataframe(input_df)

    input_columns = input_df.columns.tolist()
    if set(input_columns) != set(expected_features):
        st.error("Error: Input features do not match model's expected features.")
        st.warning(f"Expected: {expected_features}")
        st.warning(f"Input: {input_columns}")
        return
    else:
        if st.button("Predict Salary"):
            try:
                prediction = model.predict(input_df)[0]
                st.success(f"Estimated Salary: ${int(prediction):,}")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
