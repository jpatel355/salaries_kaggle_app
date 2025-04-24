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
        all_feature_names = model_dict["columns"]
        
        # Make sure we have exactly 42 features as mentioned in the error
        feature_names = all_feature_names[:42]  # Keep only the first 42 features
        
        st.sidebar.success("Model loaded successfully!")
        st.sidebar.write(f"Model type: {type(model).__name__}")
        st.sidebar.write(f"Available features: {len(all_feature_names)}")
        st.sidebar.write(f"Using features: {len(feature_names)}")
        
        # Display the feature we're removing (if any)
        if len(all_feature_names) > 42:
            removed_features = all_feature_names[42:]
            st.sidebar.warning(f"Removing extra feature: {removed_features}")
        
    except FileNotFoundError:
        st.error("Error: Model file 'Salary2022_model.pkl' not found. Make sure it's in the same directory.")
        return
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Input Widgets
    st.sidebar.header("Input Features")
    
    # Programming languages
    st.sidebar.subheader("Programming Languages")
    codes_java = st.sidebar.checkbox("I code in Java")
    codes_python = st.sidebar.checkbox("I code in Python")
    codes_sql = st.sidebar.checkbox("I code in SQL", value=True)
    codes_go = st.sidebar.checkbox("I code in GO")
    
    # Years coding and education
    years_coding = st.sidebar.slider("Years of Coding Experience", 0, 30, 5)
    education = st.sidebar.selectbox("Education Level", [0, 1, 2, 3, 4, 5])
    
    # Country
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
    
    # Handle Country as one-hot encoded (if in the feature names)
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
            # Double-check feature count
            if input_data.shape[1] != 42:
                st.error(f"Feature count mismatch! Need 42 features but have {input_data.shape[1]}.")
                # If too many features, drop extras
                if input_data.shape[1] > 42:
                    input_data = input_data.iloc[:, :42]
                    st.info(f"Dropped extra features. Now have {input_data.shape[1]} features.")
                # If too few features, this will fail, but at least we've checked
            
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
            
            # Debug information
            st.write("Input data shape:", input_data.shape)
            if hasattr(model, 'n_features_in_'):
                st.write("Model expected features:", model.n_features_in_)
            
            st.info("Try inspecting the model using debug features to ensure exact compatibility.")

    # Add a debug section
    if st.sidebar.checkbox("Debug Mode"):
        st.subheader("Debug Information")
        
        if st.button("Inspect Model"):
            try:
                # Show model attributes
                if hasattr(model, 'n_features_in_'):
                    st.write("Model expected feature count:", model.n_features_in_)
                
                if hasattr(model, 'feature_names_in_'):
                    st.write("Model expected feature names:", model.feature_names_in_)
                
                # Try to get feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    st.write("Top 10 important features:")
                    for i in range(min(10, len(indices))):
                        st.write(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
            except Exception as debug_error:
                st.error(f"Debug error: {debug_error}")

if __name__ == "__main__":
    main()
