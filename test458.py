import streamlit as st
import pandas as pd
import pickle
import numpy as np

def main():
    st.title("Salary Prediction App (Kaggle Survey 2022)")
    
    # 1. Load the pickled object
    try:
        with open("Salary2022_model.pkl", "rb") as f:
            pickled_object = pickle.load(f)
            
        # Inspect what was loaded
        st.sidebar.subheader("Model Information")
        model_type = type(pickled_object).__name__
        st.sidebar.write(f"Loaded object type: {model_type}")
        
        # If it's a dictionary, display its keys
        if isinstance(pickled_object, dict):
            st.sidebar.write("Dictionary keys:")
            st.sidebar.write(list(pickled_object.keys()))
            
            # Check if the dictionary contains a model
            if 'model' in pickled_object:
                model = pickled_object['model']
                st.sidebar.write(f"Model found in dictionary: {type(model).__name__}")
            else:
                st.warning("The loaded pickle file contains a dictionary but no direct model object. Using the dictionary for manual prediction.")
                model = pickled_object  # We'll handle prediction manually
        else:
            model = pickled_object  # Assume it's a proper model
            
    except FileNotFoundError:
        st.error("Error: Model file 'Salary2022_model.pkl' not found. Make sure it's in the same directory.")
        return
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
        
    # 2. Input Widgets
    st.sidebar.header("Input Features")
    age = st.sidebar.selectbox("Select Age Range",
                             ["18-21", "22-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-69", "70+"])
    country = st.sidebar.selectbox("Country", ["India", "US", "Spain", "Other"])
    student_status = st.sidebar.selectbox("Are you a student?", ["Yes", "No"])
    codes_python = st.sidebar.checkbox("I code in Python")
    codes_sql = st.sidebar.checkbox("I code in SQL")
    codes_java = st.sidebar.checkbox("I code in Java")
    codes_go = st.sidebar.checkbox("I code in Go")
    years_coding = st.sidebar.slider("Years of Coding Experience", 0, 50, 2)
    education = st.sidebar.selectbox("Education", [0, 1, 2, 3, 4])
    
    # 3. Process Inputs
    #  Convert age range to a numeric age (midpoint of the range)
    age_mapping = {
        "18-21": 19.5,
        "22-24": 23,
        "25-29": 27,
        "30-34": 32,
        "35-39": 37,
        "40-44": 42,
        "45-49": 47,
        "50-54": 52,
        "55-59": 57,
        "60-69": 64.5,
        "70+": 75,
    }
    age_numeric = age_mapping[age]
    
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Age': [age_numeric],
        'Years_Coding': [years_coding],
        'Education': [education],
        'Student': [1 if student_status == "Yes" else 0],
        'Country_India': [1 if country == "India" else 0],
        'Country_US': [1 if country == "US" else 0],
        'Country_Spain': [1 if country == "Spain" else 0],
        'Country_Other': [1 if country == "Other" else 0],
        'Python': [1 if codes_python else 0],
        'SQL': [1 if codes_sql else 0],
        'Java': [1 if codes_java else 0],
        'Go': [1 if codes_go else 0]
    })
    
    # Display the input data for verification
    if st.checkbox("Show input data"):
        st.write(input_data)
    
    # 4. Make Prediction
    if st.button("Predict Salary"):
        try:
            # If we have a standard model with predict method
            if hasattr(model, 'predict'):
                prediction = model.predict(input_data)
                st.success(f"Predicted Salary: ${prediction[0]:.2f}")
            # If we have a dictionary with coefficients (linear regression case)
            elif isinstance(model, dict) and 'coefficients' in model and 'intercept' in model:
                # Manual prediction for linear regression
                features = input_data.values[0]  # Get the first row as array
                coeffs = np.array(model['coefficients'])
                intercept = model['intercept']
                
                # Show which features are being used
                st.write("Using features for prediction:")
                for i, coef in enumerate(coeffs):
                    st.write(f"Feature {i}: {coef}")
                
                prediction = np.dot(features, coeffs) + intercept
                st.success(f"Predicted Salary: ${prediction:.2f}")
            # Other dictionary format
            elif isinstance(model, dict):
                st.info("Dictionary model structure detected. Please check the model format.")
                st.json(model)  # Display the dictionary structure
                st.error("Could not automatically determine how to use this dictionary for prediction.")
            else:
                st.error(f"Unknown model type: {type(model)}. Cannot make predictions.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("Make sure the model expects the same features that you're providing.")

if __name__ == "__main__":
    main()
