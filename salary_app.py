import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the model
try:
    with open("kaggle2022_model (2).pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    # It's crucial to stop here if the model fails to load
    st.stop()

# Define expected features.  This should match the features the model was trained on *exactly*.
expected_features = [
    "Age",
    "Country_India", "Country_Spain", "Country_US", "Country_Other",
    "Student_Status",
    "Codes_In_Python", "Codes_In_SQL", "Codes_In_JAVA", "Codes_In_GO",
    "Years_Coding"
]

# Create Streamlit app
def main():
    st.title("💼 Salary Prediction App (Kaggle Survey 2022)")

    # Sidebar for model explanation
    st.sidebar.markdown("### ℹ️ Model Information")
    st.sidebar.markdown(
        "This app predicts salary based on several factors.  "
        "The model was trained on data from the Kaggle 2022 survey."
    )
    st.sidebar.markdown("### 🔍 Expected Features:")
    st.sidebar.write(expected_features)  # Display the expected features in the sidebar

    # User Inputs
    age = st.selectbox("Select Age Range", ["18-21", "22-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-69", "70+"])
    country = st.selectbox("Country", ["India", "US", "Spain", "Other"])
    student_status = st.selectbox("Are you a student?", ["Yes", "No"])
    codes_python = st.checkbox("I code in Python")
    codes_sql = st.checkbox("I code in SQL")
    codes_java = st.checkbox("I code in Java")
    codes_go = st.checkbox("I code in Go")
    years_coding = st.slider("Years of Coding Experience", 0, 50, 2)

    # Convert age range to a numerical value (e.g., lower bound)
    age_mapping = {
        "18-21": 18, "22-24": 22, "25-29": 25, "30-34": 30, "35-39": 35,
        "40-44": 40, "45-49": 45, "50-54": 50, "55-59": 55, "60-69": 60, "70+": 70
    }
    age_num = age_mapping[age]

    # Prepare input data as a dictionary
    input_data = {
        "Age": age_num,  # Use the numerical age
        "Country_India": 1 if country == "India" else 0,
        "Country_Spain": 1 if country == "Spain" else 0,
        "Country_US": 1 if country == "US" else 0,
        "Country_Other": 1 if country == "Other" else 0,
        "Student_Status": 1 if student_status == "Yes" else 0,
        "Codes_In_Python": int(codes_python),
        "Codes_In_SQL": int(codes_sql),
        "Codes_In_JAVA": int(codes_java),
        "Codes_In_GO": int(codes_go),
        "Years_Coding": years_coding,
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Debugging: Print the input DataFrame to Streamlit
    st.write("📊 Input Data")
    st.dataframe(input_df)

    # Check if the input features match the expected features
    input_columns = input_df.columns.tolist()
    if set(input_columns) != set(expected_features):
        st.error("❌ Error: Input features do not match the model's expected features.")
        st.warning(f"Expected features: {expected_features}")
        st.warning(f"Input features: {input_columns}")
    else:
        # Predict salary
        if st.button("Predict Salary"): # Only predict when button is clicked.
            try:
                prediction = model.predict(input_df)[0]
                st.success(f"💰 Estimated Salary: ${int(prediction):,}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
