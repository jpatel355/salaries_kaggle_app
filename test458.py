import streamlit as st
import pandas as pd
import pickle
import numpy as np

def main():
    st.title("Salary Prediction App (Kaggle Survey 2022)")
    st.subheader(" Predict your salary based on skills, experience, and education")

    # Define the education mapping (as in the salary2025 example)
    education_mapping = {'HS': 0, 'BS': 1, 'MS': 2, 'PHD': 3}

    # Load the trained regression model
    try:
        with open("Salary2022_model.pkl", "rb") as f:
            model_dict = pickle.load(f)
            model = model_dict["model"]
            all_feature_names = model_dict["columns"]
            expected_features = 42
            feature_names = all_feature_names[:expected_features]

            st.sidebar.success("Model loaded successfully!")
            st.sidebar.write(f"Model type: {type(model).__name__}")
            st.sidebar.write(f"Available features in model file: {len(all_feature_names)}")
            st.sidebar.write(f"Using features for prediction: {len(feature_names)}")

            if len(all_feature_names) > expected_features:
                removed_features = all_feature_names[expected_features:]
                st.sidebar.warning(f"Not using these extra features: {removed_features}")
  st.sidebar.subheader("Expected Features:")
            st.sidebar.write(all_feature_names)
    except FileNotFoundError:
        st.error("Error: Model file 'Salary2022_model.pkl' not found.")
        return
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Input Widgets
    st.sidebar.header("Input Features")
    st.sidebar.subheader("Education and Experience")
    education_str = st.sidebar.selectbox("Education Level", list(education_mapping.keys()))
    years_coding = st.sidebar.slider("Years of Coding Experience", 0, 40, 5)

    st.sidebar.subheader("Programming Languages")
    codes_java = st.sidebar.checkbox("I code in Java")
    codes_python = st.sidebar.checkbox("I code in Python")
    codes_sql = st.sidebar.checkbox("I code in SQL", value=True)
    codes_go = st.sidebar.checkbox("I code in GO")

    st.sidebar.subheader("Country")
    country = st.sidebar.selectbox("Country", ["India", "US", "Canada", "Spain", "Other"])

    # Map the selected education level to its numeric value
    education_num = education_mapping[education_str]

    # Build the feature dictionary for prediction, aligning with the salary2025 example
    features = {
        "Education": education_num,
        "Years_Coding": years_coding,
        "Codes_In_JAVA": int(codes_java),
        "Codes_In_Python": int(codes_python),
        "Codes_In_SQL": int(codes_sql),
        "Codes_In_GO": int(codes_go),
        "Country_India": 0,
        "Country_Other": 0,
        "Country_Spain": 0,
        "Country_US": 0,
    }

    # Set the appropriate dummy variable for country
    if country != "Canada":
        if country == "India":
            features["Country_India"] = 1
        elif country == "US":
            features["Country_US"] = 1
        elif country == "Spain":
            features["Country_Spain"] = 1
        elif country == "Other":
            features["Country_Other"] = 1

    # Create a DataFrame from the feature dictionary
    input_data = pd.DataFrame([features])

    if st.checkbox("Show input features"):
        st.write("Input Data:")
        st.write(input_data)
        st.info(f"Total features in input: {input_data.shape[1]}")

    if st.button("Predict Salary"):
        try:
            if input_data.shape[1] != len(features): # Use the length of our created feature dict
                st.error(f"Error: Expected {len(features)} features, got {input_data.shape[1]}.")
            else:
                prediction = model.predict(input_data)[0]
                st.success(f" Estimated Salary: **${prediction:,.2f}**")

                if prediction < 40000:
                    st.info("Likely entry-level.")
                elif prediction < 80000:
                    st.info("Likely mid-level.")
                elif prediction < 120000:
                    st.info("Likely senior-level.")
                else:
                    st.info("Likely expert/leadership.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.write("Input data shape:", input_data.shape)
            if hasattr(model, 'n_features_in_'):
                st.write("Model expected features:", model.n_features_in_)
            st.info("Check error and ensure input matches model expectations.")

    st.markdown("---")
    st.markdown(
        "<small>&#9889; Built with ❤️ using Streamlit — by Your AI Assistant</small>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
