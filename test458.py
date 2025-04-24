import streamlit as st
import pandas as pd
import pickle
import numpy as np

def main():
    st.title("Salary Prediction App (Kaggle Survey 2022)")
    
    # Load the pickled dictionary
    try:
        with open("Salary2022_model.pkl", "rb") as f:
            model_dict = pickle.load(f)
            
        # Extract the model and feature names
        model = model_dict["model"]
        feature_names = model_dict["columns"]
        
        st.sidebar.success("Model loaded successfully!")
        st.sidebar.write(f"Model type: {type(model).__name__}")
        st.sidebar.write(f"Number of features: {len(feature_names)}")
        
    except FileNotFoundError:
        st.error("Error: Model file 'Salary2022_model.pkl' not found. Make sure it's in the same directory.")
        return
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Input Widgets for the first few feature names we saw
    st.sidebar.header("Input Features")
    
    # Programming languages
    st.sidebar.subheader("Programming Languages")
    codes_java = st.sidebar.checkbox("I code in Java")
    codes_python = st.sidebar.checkbox("I code in Python")
    codes_sql = st.sidebar.checkbox("I code in SQL", value=True)
    codes_go = st.sidebar.checkbox("I code in Go")
    
    # Years coding and education
    years_coding = st.sidebar.slider("Years of Coding Experience", 0, 30, 5)
    education = st.sidebar.selectbox("Education Level", [0, 1, 2, 3, 4, 5])
    country = st.sidebar.selectbox("Country", ["India", "US", "Other"])
    
    # Create a feature dictionary with exactly the expected features
    feature_dict = {}
    
    # Add the features we know about
    feature_dict["Codes_In_JAVA"] = int(codes_java)
    feature_dict["Codes_In_Python"] = int(codes_python)
    feature_dict["Codes_In_SQL"] = int(codes_sql)
    feature_dict["Codes_In_GO"] = int(codes_go)
    feature_dict["Years_Coding"] = years_coding
    feature_dict["Education"] = education
    
    # Handle Country as one-hot encoded
    if "Country_India" in feature_names:
        feature_dict["Country_India"] = 1 if country == "India" else 0
    if "Country_US" in feature_names:
        feature_dict["Country_US"] = 1 if country == "US" else 0
    if "Country_Other" in feature_names:
        feature_dict["Country_Other"] = 1 if country == "Other" else 0
    
    # Add any other features from feature_names with default value 0
    for feature in feature_names:
        if feature not in feature_dict:
            feature_dict[feature] = 0
    
    # Create input dataframe with exactly the expected features and in the right order
    input_data = pd.DataFrame([{name: feature_dict.get(name, 0) for name in feature_names}])
    
    # Display feature information
    if st.checkbox("Show input features"):
        st.write(input_data)
        st.info(f"Total features: {input_data.shape[1]}")
    
    # Make Prediction
    if st.button("Predict Salary"):
        try:
            # Make sure we have exactly the right number of features
            if input_data.shape[1] != len(feature_names):
                st.error(f"Feature count mismatch! Need {len(feature_names)} features but have {input_data.shape[1]}.")
                
                # Show what's missing or extra
                df_cols = set(input_data.columns)
                feature_set = set(feature_names)
                
                missing = feature_set - df_cols
                extra = df_cols - feature_set
                
                if missing:
                    st.warning(f"Missing features: {missing}")
                if extra:
                    st.warning(f"Extra features: {extra}")
                    # Remove extra columns
                    input_data = input_data[list(feature_set)]
                    st.info(f"Removed extra features. Now have {input_data.shape[1]} features.")
            
            # Reindex to ensure column order matches expected order
            input_data = input_data.reindex(columns=feature_names)
            
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
