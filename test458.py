import streamlit as st
import pandas as pd
import pickle
import numpy as np

def main():
    st.title("Salary Prediction App (Kaggle Survey 2022)")
    
    # 1. Load the pickled dictionary
    try:
        with open("Salary2022_model.pkl", "rb") as f:
            model_dict = pickle.load(f)
        
        # Check what's in the dictionary
        st.sidebar.success("Model dictionary loaded successfully!")
        st.sidebar.write("Dictionary keys:")
        dict_keys = list(model_dict.keys())
        st.sidebar.write(dict_keys)
        
        # The dictionary likely contains the model and feature names
        # Let's look for common patterns
        if 'model' in model_dict:
            model = model_dict['model']
            st.sidebar.write(f"Found model of type: {type(model).__name__}")
        else:
            st.sidebar.warning("No direct model found in dictionary")
            model = None
            
        # Check if feature names are stored
        feature_names = None
        possible_feature_keys = ['feature_names', 'features', 'columns', 'column_names']
        for key in possible_feature_keys:
            if key in model_dict:
                feature_names = model_dict[key]
                st.sidebar.write(f"Found feature names under key: '{key}'")
                break
                
        if feature_names is None and model is not None:
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
                st.sidebar.write("Found feature names in model.feature_names_in_")
            
        # If we found feature names, use them
        if feature_names is not None:
            st.sidebar.info(f"Number of expected features: {len(feature_names)}")
            st.sidebar.write("First few feature names:")
            st.sidebar.write(feature_names[:5])
        else:
            st.sidebar.warning("Could not determine feature names from the model dictionary")
            # Assume we need 42 features as mentioned in the error message
            feature_names = [f"feature_{i}" for i in range(42)]
            
    except FileNotFoundError:
        st.error("Error: Model file 'Salary2022_model.pkl' not found. Make sure it's in the same directory.")
        return
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
        
    # 2. Input Widgets (keeping the same inputs, but we'll be more selective with feature creation)
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
    
    # Tools & Platforms
    st.sidebar.subheader("Tools & Platforms")
    uses_jupyter = st.sidebar.checkbox("Jupyter", value=True)
    uses_vscode = st.sidebar.checkbox("VS Code", value=True)
    uses_tableau = st.sidebar.checkbox("Tableau")
    uses_excel = st.sidebar.checkbox("Excel")
    
    # Company size
    company_size = st.sidebar.selectbox("Company Size", 
                                      ["0-49 employees", "50-249 employees", "250-999 employees", 
                                       "1000-9,999 employees", "10,000+ employees"])
    
    # 3. Create features based on input
    # First, establish a baseline feature dictionary with all zeros for all expected features
    feature_dict = {name: 0 for name in feature_names}
    
    # Now, try to map our inputs to the expected feature names
    # We'll use fuzzy matching logic to handle different naming conventions
    
    # Helper function to set feature values based on partial name matches
    def set_feature_value(feature_dict, keyword, value):
        matches = [name for name in feature_dict.keys() if keyword.lower() in name.lower()]
        for match in matches:
            feature_dict[match] = value
        return len(matches) > 0  # Return whether any matches were found
    
    # Age - try to match it to age-related features
    age_numeric = {
        "18-21": 20, "22-24": 23, "25-29": 27, "30-34": 32,
        "35-39": 37, "40-44": 42, "45-49": 47, "50-54": 52,
        "55-59": 57, "60-69": 65, "70+": 75
    }[age]
    
    set_feature_value(feature_dict, 'age', age_numeric)
    
    # Gender
    gender_mapping = {
        "Man": ['male', 'man'],
        "Woman": ['female', 'woman'],
        "Non-binary": ['nonbinary', 'non_binary', 'non-binary'],
        "Prefer not to say": ['prefer_not', 'noanswer']
    }
    for gender_term in gender_mapping[gender]:
        set_feature_value(feature_dict, gender_term, 1)
    
    # Country
    country_keywords = {
        "United States": ['us', 'usa', 'united_states', 'america'],
        "India": ['india'],
        "Germany": ['germany', 'deutschland'],
        "UK": ['uk', 'united_kingdom', 'great_britain'],
        "Canada": ['canada'],
        "Other": ['other']
    }
    selected_country = country_keywords.get(country, [country.lower()])
    for term in selected_country:
        set_feature_value(feature_dict, term, 1)
    
    # Education
    edu_mapping = {
        "No formal education past high school": ['high_school', 'no_degree'],
        "Some college/university study without earning a degree": ['some_college', 'no_degree'],
        "Bachelor's degree": ['bachelor', 'bs', 'ba'],
        "Master's degree": ['master', 'ms', 'ma'],
        "Doctoral degree": ['phd', 'doctorate', 'doctoral'],
        "Professional degree": ['professional', 'md', 'jd']
    }
    for edu_term in edu_mapping[education]:
        set_feature_value(feature_dict, edu_term, 1)
    
    # Employment
    emp_mapping = {
        "Employed full-time": ['full_time', 'employed'],
        "Employed part-time": ['part_time'],
        "Independent contractor/freelancer": ['freelance', 'contractor'],
        "Student": ['student'],
        "Not employed": ['unemployed', 'not_employed']
    }
    for status in emp_mapping:
        if status in employment_status:
            for term in emp_mapping[status]:
                set_feature_value(feature_dict, term, 1)
    
    # Years of experience
    set_feature_value(feature_dict, 'year', years_coding)
    set_feature_value(feature_dict, 'coding', years_coding)
    set_feature_value(feature_dict, 'experience', years_coding)
    set_feature_value(feature_dict, 'data_science', years_data_science)
    
    # Programming languages
    if codes_python:
        set_feature_value(feature_dict, 'python', 1)
    if codes_r:
        set_feature_value(feature_dict, '_r_', 1)  # Underscore to avoid matching other features containing 'r'
    if codes_sql:
        set_feature_value(feature_dict, 'sql', 1)
    if codes_java:
        set_feature_value(feature_dict, 'java', 1)
    if codes_javascript:
        set_feature_value(feature_dict, 'javascript', 1)
        
    # Tools
    if uses_jupyter:
        set_feature_value(feature_dict, 'jupyter', 1)
    if uses_vscode:
        set_feature_value(feature_dict, 'vscode', 1)
    if uses_tableau:
        set_feature_value(feature_dict, 'tableau', 1)
    if uses_excel:
        set_feature_value(feature_dict, 'excel', 1)
    
    # Company size
    company_size_mapping = {
        "0-49 employees": ['small', 'startup'],
        "50-249 employees": ['small_medium'],
        "250-999 employees": ['medium'],
        "1000-9,999 employees": ['large'],
        "10,000+ employees": ['enterprise', 'very_large']
    }
    for size_term in company_size_mapping[company_size]:
        set_feature_value(feature_dict, size_term, 1)
    
    # Create input dataframe with the exact features the model expects
    input_data = pd.DataFrame([feature_dict])
    
    # Display the input data for verification
    if st.checkbox("Show input features"):
        st.write(input_data)
        st.info(f"Total features: {input_data.shape[1]}")
    
    # 4. Make Prediction
    if st.button("Predict Salary"):
        try:
            # Try different prediction approaches based on the dict structure
            prediction = None
            
            # If we have a model object
            if model is not None and hasattr(model, 'predict'):
                prediction = model.predict(input_data)[0]
                
            # If we have a precomputed coefficient/intercept in the dictionary
            elif 'coefficients' in model_dict and 'intercept' in model_dict:
                coeffs = np.array(model_dict['coefficients'])
                intercept = model_dict['intercept']
                
                # Ensure coefficient count matches feature count
                if len(coeffs) == len(feature_names):
                    prediction = np.dot(input_data.values[0], coeffs) + intercept
                else:
                    st.error(f"Coefficient count ({len(coeffs)}) doesn't match feature count ({len(feature_names)})")
                    
            # Check if there's a predict function in the dictionary
            elif 'predict_function' in model_dict:
                prediction = model_dict['predict_function'](input_data)
                
            # If all else fails, display the dictionary contents for debugging
            if prediction is None:
                st.error("Couldn't determine how to make predictions with the provided model dictionary")
                st.write("Dictionary contents:")
                st.json({k: str(v)[:100] for k, v in model_dict.items()})  # Show truncated values
                return
                
            # Display results
            st.success(f"Predicted Annual Salary: ${prediction:,.2f}")
            
            # Add some contextual ranges
            if prediction < 40000:
                st.info("This is in the entry-level salary range.")
            elif prediction < 80000:
                st.info("This is in the mid-level salary range.")
            elif prediction < 120000:
                st.info("This is in the senior-level salary range.")
            else:
                st.info("This is in the expert/leadership salary range.")
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("Make sure the model expects the same features that you're providing.")
            
            # Additional debugging info
            st.write("Dictionary keys:", dict_keys)
            
            # If there's a 'predict' key in the dictionary, it might be a custom prediction function
            if 'predict' in model_dict:
                st.info("Found 'predict' key in dictionary. This might be a custom function or another object.")
                st.write("Type of model_dict['predict']:", type(model_dict['predict']).__name__)

if __name__ == "__main__":
    main()
