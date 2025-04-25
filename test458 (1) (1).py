import streamlit as st
import pandas as pd
import pickle
import numpy as np  # Explicitly import numpy
from sklearn.preprocessing import OrdinalEncoder  # If you use OrdinalEncoder

def main():
    # --- 1. Load the Trained Model ---
    try:
        model_filename = "salary2022_model (2).pkl"  # Consistent filename
        with open(model_filename, "rb") as f:
            model = pickle.load(f)
        st.sidebar.success("✅ Model loaded successfully!")
    except FileNotFoundError:
        st.error(f"File not found: '{model_filename}' 📂")
        return
    except Exception as e:
        st.error(f"Error loading model: {e} 🤕")
        return

    # --- 2. Define Constants and Mappings ---
    st.title("💼 Salary Predictor")
    st.subheader("📈 Predict your salary based on skills, experience, and education")

    education_categories = ['HS', 'BS', 'MS', 'PHD']  # Consistent categories
    education_mapping = {level: i for i, level in enumerate(education_categories)}
    education_options = list(education_mapping.keys())

    # --- 2b.  Define Expected Features (CRITICAL) ---
    expected_feature_names = [
        "Codes_In_JAVA", "Codes_In_Python", "Codes_In_SQL", "Codes_In_GO", "Years_Coding", "Education",
        "Country_India", "Country_US", "Country_Spain", "Country_Canada", "Country_Germany",
        "Country_France", "Country_Japan", "Country_UK", "Country_Australia", "Country_Brazil",
        "Country_Russia", "Country_China", "Country_South Africa", "Country_Netherlands",
        "Country_Poland", "Country_Italy", "Country_Sweden", "Country_Switzerland", "Country_Turkey",
        "Country_Mexico", "Country_Argentina", "Country_Colombia", "Country_Indonesia",
        "Country_Pakistan", "Country_Egypt", "Country_Nigeria", "Country_Iran", "Country_Israel",
        "Country_Singapore", "Country_Malaysia", "Country_Philippines", "Country_Ireland",
        "Country_Belgium", "Country_Chile", "Country_Denmark", "Country_Finland"
    ]
    num_expected_features = len(expected_feature_names)
    all_countries = ["India", "US", "Canada", "Spain", "Germany", "France", "Japan", "UK", "Australia", "Brazil", "Russia", "China", "South Africa", "Netherlands", "Poland", "Italy", "Sweden", "Switzerland", "Turkey", "Mexico", "Argentina", "Colombia", "Indonesia", "Pakistan", "Egypt", "Nigeria", "Iran", "Israel", "Singapore", "Malaysia", "Philippines", "Ireland", "Belgium", "Chile", "Denmark", "Finland", "Norway", "Other"]

    # --- 3. User Input Widgets ---
    with st.sidebar:  # Group sidebar elements
        st.header("Input Features")
        education = st.selectbox("Education Level", education_options)
        years_coding = st.slider("⏳ Years of Coding Experience", 0, 40, 5)
        country = st.selectbox("🌎 Country", all_countries)
        codes_java = st.checkbox("☕ I code in Java")
        codes_python = st.checkbox("🐍 I code in Python")
        codes_sql = st.checkbox("📊 I code in SQL")
        codes_go = st.checkbox("🐹 I code in GO")

    # --- 4. Prepare Input Data for Prediction ---
    def prepare_input_data(education, years_coding, country, codes_java, codes_python, codes_sql, codes_go):
        """
        Prepares the input data as a DataFrame that matches the model's expected format.
        """

        # Initialize all features to 0
        features = {feature: 0 for feature in expected_feature_names}

        # Assign basic features
        features["Education"] = education_mapping[education]
        features["Years_Coding"] = years_coding
        features["Codes_In_JAVA"] = int(codes_java)
        features["Codes_In_Python"] = int(codes_python)
        features["Codes_In_SQL"] = int(codes_sql)
        features["Codes_In_GO"] = int(codes_go)

        # One-hot encode country
        country_column = f"Country_{country}"
        if country_column in features:
            features[country_column] = 1
        elif "Country_Other" in features and country not in [c.split("_")[1] if "_" in c and c.startswith("Country_") else "" for c in expected_feature_names]:
            features["Country_Other"] = 1
        else:
            st.warning(f"Selected country '{country}' not found in model's training data. Using 'Other'.")
            features["Country_Other"] = 1  # Default to 'Other'

        return pd.DataFrame([features], columns=expected_feature_names)

    input_df = prepare_input_data(education, years_coding, country, codes_java, codes_python, codes_sql, codes_go)

    # --- 5. Debugging (Optional) ---
    if st.checkbox("🔍 Show input features"):
        st.write("Input Data:")
        st.dataframe(input_df)  # Use st.dataframe for better display
        st.info(f"Number of features in input data: {input_df.shape[1]}")
        st.info(f"Number of expected features: {num_expected_features}")
        st.info(f"Columns in input data: {input_df.columns.tolist()}")
        st.info(f"Expected columns: {expected_feature_names}")

    # --- 6. Make Prediction and Display Result ---
    if st.button("💰 Predict Salary"):
        try:
            if input_df.shape[1] == num_expected_features:
                prediction = model.predict(input_df)[0]
                st.success(f"🎉 Estimated Annual Salary: ${prediction:,.2f}")
            else:
                st.error(f"Error: Input data has {input_df.shape[1]} features, but the model expects {num_expected_features}.")
        except Exception as e:
            st.error(f"Error making prediction: {e} 🤕")

    # --- 7. Footer ---
    st.markdown("---")
    st.markdown("<small>✨ Built with ❤️ using Streamlit — Jiya, Rhea, and Michael</small>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
