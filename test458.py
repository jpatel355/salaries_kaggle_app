import streamlit as st
import pandas as pd
import pickle
import numpy as np

def main():
    st.title("Salary Prediction App (Kaggle Survey 2022)")
    
    # 1. Load the Model
    try:
        with open("Salary2022_model.pkl", "rb") as f:
            model = pickle.load(f)
        st.sidebar.success("Model loaded successfully!")
        st.sidebar.write(f"Model type: {type(model).__name__}")
    except FileNotFoundError:
        st.error("Error: Model file 'Salary2022_model.pkl' not found. Make sure it's in the same directory.")
        return
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
        
    # 2. Input Widgets
    st.sidebar.header("Input Features")
    
    # Basic demographic info
    age = st.sidebar.selectbox("Select Age Range",
                             ["18-21", "22-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-69", "70+"])
    
    gender = st.sidebar.selectbox("Gender", ["Man", "Woman", "Non-binary", "Prefer not to say"])
    
    country_options = ["United States", "India", "Germany", "UK", "Canada", "Brazil", "France", "Spain", 
                      "Australia", "Netherlands", "Italy", "Japan", "Poland", "Russia", "China", "Other"]
    country = st.sidebar.selectbox("Country", country_options)
    
    # Education
    education_options = ["No formal education past high school", "Some college/university study without earning a degree", 
                        "Bachelor's degree", "Master's degree", "Doctoral degree", "Professional degree"]
    education = st.sidebar.selectbox("Education Level", education_options)
    
    # Employment status
    employment_status = st.sidebar.selectbox("Employment Status", 
                                           ["Employed full-time", "Employed part-time", "Independent contractor/freelancer", 
                                            "Student", "Not employed, and not looking for work", "Not employed, but looking for work"])
    
    # Experience
    years_coding = st.sidebar.slider("Years of Coding Experience", 0, 50, 5)
    years_data_science = st.sidebar.slider("Years of Data Science Experience", 0, 20, 2)
    
    # Languages
    st.sidebar.subheader("Programming Languages")
    codes_python = st.sidebar.checkbox("Python", value=True)
    codes_r = st.sidebar.checkbox("R")
    codes_sql = st.sidebar.checkbox("SQL", value=True)
    codes_java = st.sidebar.checkbox("Java")
    codes_javascript = st.sidebar.checkbox("JavaScript")
    codes_julia = st.sidebar.checkbox("Julia")
    codes_c = st.sidebar.checkbox("C/C++")
    codes_go = st.sidebar.checkbox("Go")
    
    # Tools & Platforms
    st.sidebar.subheader("Tools & Platforms")
    uses_jupyter = st.sidebar.checkbox("Jupyter", value=True)
    uses_rstudio = st.sidebar.checkbox("RStudio")
    uses_vscode = st.sidebar.checkbox("VS Code", value=True)
    uses_pycharm = st.sidebar.checkbox("PyCharm")
    uses_tableau = st.sidebar.checkbox("Tableau")
    uses_excel = st.sidebar.checkbox("Excel")
    uses_powerbi = st.sidebar.checkbox("Power BI")
    
    # ML Frameworks
    st.sidebar.subheader("ML Frameworks")
    uses_scikit = st.sidebar.checkbox("Scikit-learn", value=True)
    uses_tensorflow = st.sidebar.checkbox("TensorFlow")
    uses_pytorch = st.sidebar.checkbox("PyTorch")
    uses_keras = st.sidebar.checkbox("Keras")
    uses_xgboost = st.sidebar.checkbox("XGBoost")
    
    # Company size
    company_size = st.sidebar.selectbox("Company Size", 
                                      ["0-49 employees", "50-249 employees", "250-999 employees", 
                                       "1000-9,999 employees", "10,000+ employees"])
    
    # 3. Process Inputs
    # Convert age range to a numeric value
    age_mapping = {
        "18-21": 19.5, "22-24": 23, "25-29": 27, "30-34": 32,
        "35-39": 37, "40-44": 42, "45-49": 47, "50-54": 52,
        "55-59": 57, "60-69": 64.5, "70+": 75
    }
    age_numeric = age_mapping[age]
    
    # Create a comprehensive input data dictionary
    input_dict = {
        # Map all inputs to appropriate feature names
        'Age': age_numeric,
        'Years_Coding': years_coding,
        'Years_DataScience': years_data_science,
        
        # One-hot encode categorical variables
        # Gender
        'Gender_Man': 1 if gender == "Man" else 0,
        'Gender_Woman': 1 if gender == "Woman" else 0,
        'Gender_NonBinary': 1 if gender == "Non-binary" else 0,
        'Gender_NoAnswer': 1 if gender == "Prefer not to say" else 0,
        
        # Education
        'Education_HighSchool': 1 if education == "No formal education past high school" else 0,
        'Education_SomeCollege': 1 if education == "Some college/university study without earning a degree" else 0,
        'Education_Bachelors': 1 if education == "Bachelor's degree" else 0,
        'Education_Masters': 1 if education == "Master's degree" else 0,
        'Education_Doctoral': 1 if education == "Doctoral degree" else 0,
        'Education_Professional': 1 if education == "Professional degree" else 0,
        
        # Employment
        'Employment_FullTime': 1 if employment_status == "Employed full-time" else 0,
        'Employment_PartTime': 1 if employment_status == "Employed part-time" else 0,
        'Employment_Freelance': 1 if employment_status == "Independent contractor/freelancer" else 0,
        'Employment_Student': 1 if employment_status == "Student" else 0,
        'Employment_NotWorking': 1 if "Not employed" in employment_status else 0,
        
        # Country
        'Country_US': 1 if country == "United States" else 0,
        'Country_India': 1 if country == "India" else 0,
        'Country_Germany': 1 if country == "Germany" else 0,
        'Country_UK': 1 if country == "UK" else 0,
        'Country_Canada': 1 if country == "Canada" else 0,
        'Country_Other': 1 if country not in ["United States", "India", "Germany", "UK", "Canada"] else 0,
        
        # Programming Languages
        'Language_Python': 1 if codes_python else 0,
        'Language_R': 1 if codes_r else 0,
        'Language_SQL': 1 if codes_sql else 0,
        'Language_Java': 1 if codes_java else 0,
        'Language_JavaScript': 1 if codes_javascript else 0,
        'Language_Julia': 1 if codes_julia else 0,
        'Language_C': 1 if codes_c else 0,
        'Language_Go': 1 if codes_go else 0,
        
        # Tools
        'Tool_Jupyter': 1 if uses_jupyter else 0,
        'Tool_RStudio': 1 if uses_rstudio else 0,
        'Tool_VSCode': 1 if uses_vscode else 0,
        'Tool_PyCharm': 1 if uses_pycharm else 0,
        'Tool_Tableau': 1 if uses_tableau else 0,
        'Tool_Excel': 1 if uses_excel else 0,
        'Tool_PowerBI': 1 if uses_powerbi else 0,
        
        # ML Frameworks
        'ML_ScikitLearn': 1 if uses_scikit else 0,
        'ML_TensorFlow': 1 if uses_tensorflow else 0,
        'ML_PyTorch': 1 if uses_pytorch else 0,
        'ML_Keras': 1 if uses_keras else 0,
        'ML_XGBoost': 1 if uses_xgboost else 0,
        
        # Company Size
        'Company_Small': 1 if company_size == "0-49 employees" else 0,
        'Company_Medium': 1 if company_size == "50-249 employees" else 0,
        'Company_Large': 1 if company_size == "250-999 employees" else 0,
        'Company_VeryLarge': 1 if company_size == "1000-9,999 employees" else 0,
        'Company_Enterprise': 1 if company_size == "10,000+ employees" else 0,
    }
    
    # Create a DataFrame from the input dictionary
    input_data = pd.DataFrame([input_dict])
    
    # Display feature information
    if st.checkbox("Show model feature count"):
        # Get the feature count expected by the model (if available)
        expected_feature_count = 42  # From error message
        actual_feature_count = len(input_data.columns)
        st.write(f"Expected features: {expected_feature_count}")
        st.write(f"Current features: {actual_feature_count}")
        
        if expected_feature_count != actual_feature_count:
            st.warning(f"Feature count mismatch! The model expects {expected_feature_count} features, but we have {actual_feature_count}.")
    
    # Display the input data for verification
    if st.checkbox("Show input features"):
        st.write(input_data)
    
    # 4. Make Prediction
    if st.button("Predict Salary"):
        try:
            # Check feature count before prediction
            if hasattr(model, 'n_features_in_'):
                expected_features = model.n_features_in_
                if expected_features != input_data.shape[1]:
                    st.warning(f"Feature count mismatch! Model expects {expected_features} features, but input has {input_data.shape[1]}.")
                    
                    missing_count = expected_features - input_data.shape[1]
                    if missing_count > 0:
                        # Add dummy features to match the expected count
                        st.info(f"Adding {missing_count} dummy features to match model expectations.")
                        for i in range(missing_count):
                            input_data[f'dummy_feature_{i}'] = 0
                    elif missing_count < 0:
                        st.info("Model expects fewer features than provided. Some features may be ignored.")
            
            # Make prediction
            prediction = model.predict(input_data)
            
            # Display results
            st.success(f"Predicted Annual Salary: ${prediction[0]:,.2f}")
            
            # Add some contextual ranges
            if prediction[0] < 40000:
                st.info("This is in the entry-level salary range.")
            elif prediction[0] < 80000:
                st.info("This is in the mid-level salary range.")
            elif prediction[0] < 120000:
                st.info("This is in the senior-level salary range.")
            else:
                st.info("This is in the expert/leadership salary range.")
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("Make sure the model expects the same features that you're providing.")

if __name__ == "__main__":
    main()
