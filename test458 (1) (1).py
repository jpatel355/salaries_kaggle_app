
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
        
        # Debug model information
        if st.checkbox("Show Model Information"):
            st.write("Model type:", type(model))
            if hasattr(model, 'feature_names_in_'):
                st.write("Model features:", model.feature_names_in_)
                st.write("Number of features expected:", len(model.feature_names_in_))
            elif hasattr(model, 'n_features_in_'):
                st.write("Number of features expected:", model.n_features_in_)
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

    # --- Get all expected features from the model ---
    if hasattr(model, 'feature_names_in_'):
        expected_feature_names = model.feature_names_in_.tolist()
        # Extract the list of countries from the feature names
        all_countries = [country.split('_')[1] for country in expected_feature_names 
                        if country.startswith('Country_')]
        if 'Country_Other' in expected_feature_names:
            all_countries.append('Other')
    else:
        # Fallback with a much larger set of potential features
        all_countries = ["India", "US", "Canada", "Spain", "Germany", "France", "Japan", "UK", "Australia", "Brazil", 
                        "Russia", "China", "South Africa", "Netherlands", "Poland", "Italy", "Sweden", "Switzerland", 
                        "Turkey", "Mexico", "Argentina", "Colombia", "Indonesia", "Pakistan", "Egypt", "Nigeria", 
                        "Iran", "Israel", "Singapore", "Malaysia", "Philippines", "Ireland", "Belgium", "Chile", 
                        "Denmark", "Finland", "Norway", "Other"]
        
        # Construct a comprehensive list of likely features
        # This is a larger set to try to match the 81 features the model expects
        programming_languages = ["JAVA", "Python", "SQL", "GO", "JavaScript", "TypeScript", "C", "C++", "C#", "PHP", 
                               "Ruby", "Swift", "Kotlin", "Rust", "Scala", "R", "MATLAB", "Shell", "Assembly", "Perl"]
        
        expected_feature_names = []
        # Add programming language features
        for lang in programming_languages:
            expected_feature_names.append(f"Codes_In_{lang}")
        
        # Add basic features
        expected_feature_names.extend(["Years_Coding", "Education"])
        
        # Add country features
        for country in all_countries:
            if country != "Other":  # Skip "Other" as it might be handled differently
                expected_feature_names.append(f"Country_{country}")

    num_expected_features = len(expected_feature_names)
    st.sidebar.info(f"Model expects {num_expected_features} features")

    # --- 3. User Input Widgets ---
    with st.sidebar:
        st.header("Enter Your Details")
        education = st.selectbox("Education Level", education_options)
        years_coding = st.slider("⏳ Years of Coding", 0, 40, 5)
        country = st.selectbox("🌎 Country", all_countries)
        
        # Dynamic programming language checkboxes based on the model's expected features
        programming_languages = [feature.replace("Codes_In_", "") for feature in expected_feature_names if feature.startswith("Codes_In_")]
        coding_skills = {}
        
        if programming_languages:
            st.subheader("Programming Languages")
            for lang in programming_languages:
                emoji = {
                    "JAVA": "☕", "Python": "🐍", "SQL": "📊", "GO": "🐹", 
                    "JavaScript": "📜", "TypeScript": "🔷", "C": "🔣", "C++": "🧮", 
                    "C#": "🎮", "PHP": "🐘", "Ruby": "💎", "Swift": "🍎", 
                    "Kotlin": "🤖", "Rust": "⚙️", "Scala": "📈", "R": "📊", 
                    "MATLAB": "🧮", "Shell": "🐚", "Assembly": "⚡", "Perl": "🐪"
                }.get(lang, "💻")
                coding_skills[lang] = st.checkbox(f"{emoji} Codes in {lang}")
        else:
            # Fallback to basic languages if no programming languages were detected
            coding_skills = {
                "JAVA": st.checkbox("☕ Codes in Java"),
                "Python": st.checkbox("🐍 Codes in Python"),
                "SQL": st.checkbox("📊 Codes in SQL"),
                "GO": st.checkbox("🐹 Codes in Go")
            }

    # --- 4. Prepare Input Data Function ---
    def prepare_input_data(education, years_coding, country, coding_skills, expected_features):
        """
        Prepares the input data for prediction, handling one-hot encoding and ensuring correct feature order.
        """
        features = {feature: 0 for feature in expected_features}

        # Assign basic features
        if "Education" in features:
            features["Education"] = education_mapping[education]
        if "Years_Coding" in features:
            features["Years_Coding"] = years_coding
        
        # Assign programming language features
        for lang, has_skill in coding_skills.items():
            feature_name = f"Codes_In_{lang}"
            if feature_name in features:
                features[feature_name] = int(has_skill)
        
        # One-hot encode country
        country_column = f"Country_{country}"
        if country_column in features:
            features[country_column] = 1
        elif "Country_Other" in features and country == "Other":
            features["Country_Other"] = 1
        elif "Country_Other" in features:
            st.warning(f"Warning: Country '{country}' not found in model features. Using 'Other' instead.")
            features["Country_Other"] = 1
        
        # Create DataFrame with features in exact order expected by model
        input_df = pd.DataFrame([features], columns=expected_features)
        return input_df

    # --- 5. Prepare Input DataFrame ---
    input_df = prepare_input_data(education, years_coding, country, coding_skills, expected_feature_names)

    # --- 6. Debugging (Show Input Data) ---
    if st.checkbox("Show Input Data for Debugging"):
        st.subheader("Input Data for Prediction")
        st.dataframe(input_df)
        st.text(f"Input Data Shape: {input_df.shape}")
        st.text(f"Input Data Columns: {input_df.columns.tolist()}")
        st.text(f"Expected Features: {expected_feature_names}")

        # Column comparison for debugging
        if hasattr(model, 'feature_names_in_'):
            model_columns = model.feature_names_in_.tolist()
            input_columns = input_df.columns.tolist()
            
            missing_in_input = set(model_columns) - set(input_columns)
            missing_in_model = set(input_columns) - set(model_columns)
            
            if missing_in_input:
                st.error(f"ERROR: Columns missing in input data: {missing_in_input}")
            if missing_in_model:
                st.error(f"ERROR: Columns in input not expected by model: {missing_in_model}")
            
            if model_columns != input_columns:
                st.warning("Warning: Order of columns may not match model expectations")

    # --- 7. Make Prediction ---
    if st.button("💰 Predict Salary"):
        try:
            # Check if input data has correct number of features
            model_features = len(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else model.n_features_in_
            
            if input_df.shape[1] != model_features:
                st.error(f"ERROR: Input data has {input_df.shape[1]} features, but the model expects {model_features}. Please check input values and model file.")
                st.info("Try using the 'Show Model Information' and 'Show Input Data for Debugging' options to identify the issue.")
            else:
                # Make sure the column order matches exactly
                if hasattr(model, 'feature_names_in_'):
                    # Reorder columns to match model expectations
                    input_df = input_df[model.feature_names_in_]
                
                prediction = model.predict(input_df)[0]
                st.success(f"🎉 Estimated Annual Salary: ${prediction:,.2f}")
        except Exception as e:
            st.error(f"ERROR: An unexpected error occurred during prediction: {e}")
            st.info("If the error persists, try using the debugging tools to identify missing or mismatched features.")

    # --- 8. Footer ---
    st.markdown("---")
    st.markdown("Developed with ❤️ by Jiya, Rhea, and Michael", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
