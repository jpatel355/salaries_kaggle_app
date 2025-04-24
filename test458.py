import streamlit as st
import pandas as pd
import pickle
import numpy as np

def main():
    st.title("Salary Prediction App (Kaggle Survey 2022)")
    st.subheader(" Predict your salary based on skills, experience, and education")

    # Define the education mapping (assuming numerical encoding is already correct)
    education_levels = [0, 1, 2, 3, 4, 5] # Assuming these correspond to the numerical values

    # List of the 42 features the model expects
    feature_names = [
        "Codes_In_JAVA", "Codes_In_Python", "Codes_In_SQL", "Codes_In_GO", "Years_Coding", "Education",
        "Country_Australia", "Country_Bangladesh", "Country_Brazil", "Country_Canada", "Country_Chile",
        "Country_China", "Country_Colombia", "Country_Egypt", "Country_France", "Country_Ghana",
        "Country_India", "Country_Indonesia", "Country_Iran, Islamic Republic of...", "Country_Israel",
        "Country_Italy", "Country_Japan", "Country_Kenya", "Country_Mexico", "Country_Morocco",
        "Country_Netherlands", "Country_Nigeria", "Country_Other", "Country_Pakistan", "Country_Peru",
        "Country_Philippines", "Country_Poland", "Country_Russia", "Country_South Africa",
        "Country_South Korea", "Country_Spain", "Country_Taiwan", "Country_Thailand", "Country_Tunisia",
        "Country_Turkey", "Country_United Kingdom of Great Britain and Northern Ireland",
        "Country_United States of America"
    ]
    expected_features = len(feature_names)

    # Load the trained regression model
    try:
        with open("Salary2022_model.pkl", "rb") as f:
            model_dict = pickle.load(f)
            model = model_dict["model"]
            all_features_from_model = model_dict["columns"]

            st.sidebar.success("Model loaded successfully!")
            st.sidebar.write(f"Model type: {type(model).__name__}")
            st.sidebar.write(f"Available features in model file: {len(all_features_from_model)}")
            st.sidebar.write(f"Using features for prediction: {expected_features}")
            if len(all_features_from_model) > expected_features:
                st.sidebar.warning(f"Not using extra features: {all_features_from_model[expected_features:]}")

            st.sidebar.subheader("Input Features")

            # Create input widgets based on the feature names
            codes_java = st.sidebar.checkbox("I code in Java", key="java")
            codes_python = st.sidebar.checkbox("I code in Python", key="python")
            codes_sql = st.sidebar.checkbox("I code in SQL", key="sql")
            codes_go = st.sidebar.checkbox("I code in GO", key="go")
            years_coding = st.sidebar.slider("Years of Coding Experience", 0, 30, 5, key="years")
            education = st.sidebar.selectbox("Education Level", education_levels, key="education")
            country = st.sidebar.selectbox("Country", [
                "Australia", "Bangladesh", "Brazil", "Canada", "Chile", "China", "Colombia", "Egypt",
                "France", "Ghana", "India", "Indonesia", "Iran, Islamic Republic of...", "Israel",
                "Italy", "Japan", "Kenya", "Mexico", "Morocco", "Netherlands", "Nigeria", "Other",
                "Pakistan", "Peru", "Philippines", "Poland", "Russia", "South Africa", "South Korea",
                "Spain", "Taiwan", "Thailand", "Tunisia", "Turkey",
                "United Kingdom of Great Britain and Northern Ireland", "United States of America"
            ], key="country")

            # Create the feature dictionary
            features = {}
            features["Codes_In_JAVA"] = int(codes_java)
            features["Codes_In_Python"] = int(codes_python)
            features["Codes_In_SQL"] = int(codes_sql)
            features["Codes_In_GO"] = int(codes_go)
            features["Years_Coding"] = years_coding
            features["Education"] = education

            # Handle Country one-hot encoding
            for country_name in [
                "Australia", "Bangladesh", "Brazil", "Canada", "Chile", "China", "Colombia", "Egypt",
                "France", "Ghana", "India", "Indonesia", "Iran, Islamic Republic of...", "Israel",
                "Italy", "Japan", "Kenya", "Mexico", "Morocco", "Netherlands", "Nigeria", "Other",
                "Pakistan", "Peru", "Philippines", "Poland", "Russia", "South Africa", "South Korea",
                "Spain", "Taiwan", "Thailand", "Tunisia", "Turkey",
                "United Kingdom of Great Britain and Northern Ireland", "United States of America"
            ]:
                features[f"Country_{country_name}"] = 1 if country == country_name else 0

            # Create the input DataFrame with the correct column order
            input_data = pd.DataFrame([features])

            if st.checkbox("Show input features"):
                st.write("Input Data:")
                st.write(input_data)
                st.info(f"Total features in input: {input_data.shape[1]}")

            if st.button("Predict Salary"):
                try:
                    if input_data.shape[1] != expected_features:
                        st.error(f"Error: Expected {expected_features} features, got {input_data.shape[1]}.")
                    else:
                        prediction = model.predict(input_data)
                        st.success(f" Estimated Annual Salary: ${prediction[0]:,.2f}")

                        if prediction[0] < 40000:
                            st.info("Likely entry-level.")
                        elif prediction[0] < 80000:
                            st.info("Likely mid-level.")
                        elif prediction[0] < 120000:
                            st.info("Likely senior-level.")
                        else:
                            st.info("Likely expert/leadership.")

                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.write("Input data shape:", input_data.shape)
                    if hasattr(model, 'n_features_in_'):
                        st.write("Model expected features:", model.n_features_in_)
                    st.info("Check error and ensure input matches model expectations.")

    except FileNotFoundError:
        st.error("Error: Model file 'Salary2022_model.pkl' not found.")
        return
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    st.markdown("---")
    st.markdown(
        "<small>&#9889; Built with ❤️ using Streamlit — by Your AI Assistant</small>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
