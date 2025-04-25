import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def main():
    # --- 1. Model Loading ---
    model_filename = "salary2022_model (2).pkl"
    try:
        with open(model_filename, "rb") as f:
            model = pickle.load(f)
        st.sidebar.success("✅ Model loaded successfully!")
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_filename}' not found. Please ensure it's in the correct location. 📂")
        return
    except Exception as e:
        st.error(f"Error loading model: {e} 🤕")
        return

    # --- 2. Define Constants and Data Structures ---
    st.title("💼 Salary Predictor")
    st.subheader("Accurate salary predictions based on your profile.")

    education_categories = ['HS', 'BS', 'MS', 'PHD']
    education_mapping = {level: i for i, level in enumerate(education_categories)}
    education_options = list(education_mapping.keys())

    all_countries = ["India", "US", "Canada", "Spain", "Germany", "France", "Japan", "UK", "Australia", "Brazil", "Russia", "China", "South Africa", "Netherlands", "Poland", "Italy", "Sweden", "Switzerland", "Turkey", "Mexico", "Argentina", "Colombia", "Indonesia", "Pakistan", "Egypt", "Nigeria", "Iran", "Israel", "Singapore", "Malaysia", "Philippines", "Ireland", "Belgium", "Chile", "Denmark", "Finland", "Norway", "Other"]

    # --- 2b. CRITICAL: Define Expected Features from Training ---
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

    # --- 3. User Input Widgets ---
    with st.sidebar:
        st.header("Enter Your Details")
        education = st.selectbox("Education Level", education_options)
        years_coding = st.slider("⏳ Years of Coding", 0, 40, 5)
        country = st.selectbox("🌎 Country", all_countries)
        codes_java = st.checkbox("☕ Codes in Java")
        codes_python = st.checkbox("🐍 Codes in Python")
        codes_sql = st.checkbox("📊 Codes in SQL")
        codes_go = st.checkbox("🐹 Codes in Go")

    # --- 4. Prepare Input Data Function ---
    def prepare_input_data(education, years_coding, country, codes_java, codes_python, codes_sql, codes_go, expected_features):
        """
        Prepares the input data for prediction, handling one-hot encoding and ensuring correct feature order.

        Args:
            education (str): User-selected education level.
            years_coding (int): Years of coding experience.
            country (str): User-selected country.
            codes_java (bool): Whether the user codes in Java.
            codes_python (bool): Whether the user codes in Python.
            codes_sql (bool): Whether the user codes in SQL.
            codes_go (bool): Whether the user codes in Go.
            expected_features (list): List of feature names the model expects.

        Returns:
            pd.DataFrame: Prepared input DataFrame.
        """

        features = {feature: 0 for feature in expected_features}

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
        elif "Country_Other" in features:
            if country not in [c.split("_")[1] for c in expected_features if c.startswith("Country_")]:
                features["Country_Other"] = 1
            else:
                st.warning(f"Warning: Country '{country}' was in training data but not explicitly handled. Using 'Other'")
                features["Country_Other"] = 1
        else:
            st.warning(f"Warning: Country '{country}' not found and 'Country_Other' not in training data. Prediction may be unreliable.")

        input_df = pd.DataFrame([features], columns=expected_features)
        return input_df

    # --- 5. Prepare Input DataFrame ---
    input_df = prepare_input_data(education, years_coding, country, codes_java, codes_python, codes_sql, codes_go, expected_feature_names)

    # --- 6. Debugging (Show Input Data) ---
    if st.checkbox("Show Input Data for Debugging"):
        st.subheader("Input Data for Prediction")
        st.dataframe(input_df)
        st.text(f"Input Data Shape: {input_df.shape}")
        st.text(f"Input Data Columns: {input_df.columns.tolist()}")
        st.text(f"Expected Features: {expected_feature_names}")

        # --- 6b. Even More Robust Debugging: Column Comparison ---
        def find_column_differences(list1, list2):
            set1 = set(list1)
            set2 = set(list2)
            in_list1_only = list(set1 - set2)
            in_list2_only = list(set2 - set1)
            return in_list1_only, in_list2_only

        training_cols = expected_feature_names  # Assuming expected_feature_names is from training
        input_cols = input_df.columns.tolist()

        missing_in_input, missing_in_training = find_column_differences(training_cols, input_cols)

        if missing_in_input:
            st.error(f"ERROR: Columns missing in input data: {missing_in_input}")
        if missing_in_training:
            st.error(f"ERROR: Columns missing in training data (this should not happen!): {missing_in_training}")

    # --- 7. Make Prediction ---
    if st.button("💰 Predict Salary"):
        try:
            if input_df.shape[1] != num_expected_features:
                st.error(f"ERROR: Input data has {input_df.shape[1]} features, but the model expects {num_expected_features}. Please check input values and model file.")
            else:
                prediction = model.predict(input_df)[0]
                st.success(f"🎉 Estimated Annual Salary: ${prediction:,.2f}")
        except Exception as e:
            st.error(f"ERROR: An unexpected error occurred during prediction: {e}")

    # --- 8. Footer ---
    st.markdown("---")
    st.markdown("<small>✨ Developed with ❤️ using Streamlit</small>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
