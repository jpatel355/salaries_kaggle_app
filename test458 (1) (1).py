import streamlit as st
import pandas as pd
import pickle
import numpy as np

def main():
    # --- 1. Model Loading ---
    model_filename = "salary2022_model (2).pkl"
    try:
        with open(model_filename, "rb") as f:
            model = pickle.load(f)
        st.sidebar.success("✅ Model loaded successfully!")

        if st.checkbox("Show Model Information"):
            st.write("Model type:", type(model))
            if hasattr(model, 'n_features_in_'):
                st.write("Number of features expected:", model.n_features_in_)
            else:
                st.write("Could not determine number of features expected")
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_filename}' not found. 📂")
        return
    except Exception as e:
        st.error(f"Error loading model: {e} 🤕")
        return

    # --- 2. Set up UI Elements ---
    st.title("💼 Salary Predictor")
    st.subheader("Accurate salary predictions based on your profile.")

    # --- 3. Define EXACTLY 81 Features ---
    programming_languages = [
        "JAVA", "Python", "SQL", "GO", "JavaScript", "TypeScript", "C", "C++", "C#", "PHP", "Ruby", "Swift",
        "Kotlin", "Rust", "Scala", "R", "MATLAB", "Shell", "Assembly", "Perl", "VBA", "Objective-C", "Dart",
        "Lua", "Groovy", "Haskell", "Clojure", "F#", "Julia", "COBOL"
    ]
    countries = [
        "India", "US", "Canada", "UK", "Germany", "France", "Australia", "Brazil", "Russia", "China", "Japan",
        "Spain", "Italy", "Netherlands", "Poland", "Sweden", "Switzerland", "Turkey", "Mexico", "Argentina",
        "Colombia", "Indonesia", "Pakistan", "Egypt", "Nigeria", "Iran", "Israel", "Singapore", "Malaysia",
        "Philippines", "Ireland", "Belgium", "Chile", "Denmark", "Finland", "Norway", "South Africa", "Ukraine",
        "Portugal", "Austria", "Greece", "Czech Republic", "Hungary", "Romania", "New Zealand", "Other"
    ]
    base_features = ["Education", "Years_Coding"]
    additional_features = [
        "Years_Professional_Experience", "Team_Size", "Company_Size", "Is_Manager"
    ]
    all_feature_names = [f"Codes_In_{lang}" for lang in programming_languages]
    all_feature_names.extend(base_features)
    all_feature_names.extend([f"Country_{c}" for c in countries])
    all_feature_names.extend(additional_features)

    if len(all_feature_names) < 81:
        for i in range(len(all_feature_names), 81):
            all_feature_names.append(f"Feature_{i}")
    elif len(all_feature_names) > 81:
        all_feature_names = all_feature_names[:81]

    st.sidebar.info("Using exactly 81 features")
    st.sidebar.write(f"🧮 Total Features Prepared: {len(all_feature_names)}")
    st.sidebar.write("🔍 Sample Features:", all_feature_names[:5], "...")

    # --- 4. User Input Widgets ---
    with st.sidebar:
        st.header("Enter Your Details")
        education_categories = ['HS', 'BS', 'MS', 'PHD']
        education_mapping = {level: i for i, level in enumerate(education_categories)}
        education = st.selectbox("Education Level", education_categories)
        years_coding = st.slider("⏳ Years of Coding", 0, 40, 5)

        available_countries = [c.replace("Country_", "") for c in all_feature_names if c.startswith("Country_")]
        country = st.selectbox("🌎 Country", available_countries)

        lang_features = [f for f in all_feature_names if f.startswith("Codes_In_")]
        available_languages = [l.replace("Codes_In_", "") for l in lang_features]

        st.subheader("Programming Languages")
        coding_skills = {}
        emoji_map = {
            "JAVA": "☕", "Python": "🐍", "SQL": "📊", "GO": "🐹", "JavaScript": "📜", "TypeScript": "🔷", "C": "🔣",
            "C++": "🧮", "C#": "🎮", "PHP": "🐘", "Ruby": "💎", "Swift": "🍎", "Kotlin": "🤖", "Rust": "⚙️",
            "Scala": "📈", "R": "📊", "MATLAB": "🧮", "Shell": "🐚", "Assembly": "⚡", "Perl": "🐪"
        }
        col1, col2 = st.columns(2)
        for i, lang in enumerate(available_languages):
            emoji = emoji_map.get(lang, "💻")
            if i % 2 == 0:
                coding_skills[lang] = col1.checkbox(f"{emoji} {lang}")
            else:
                coding_skills[lang] = col2.checkbox(f"{emoji} {lang}")

        additional_feature_inputs = {}
        additional_feature_inputs["Years_Professional_Experience"] = st.slider(
            "Years of Professional Experience", 0, 40, 3)
        additional_feature_inputs["Team_Size"] = st.slider("Team Size", 1, 100, 10)
        company_size = st.selectbox("Company Size", ["< 10", "10-50", "50-250", "250-1000", "> 1000"], index=2)
        company_size_map = {"< 10": 0, "10-50": 1, "50-250": 2, "250-1000": 3, "> 1000": 4}
        additional_feature_inputs["Company_Size"] = company_size_map[company_size]
        additional_feature_inputs["Is_Manager"] = st.checkbox("Is a Manager")

    # --- 5. Prepare Input Data Function ---
    def prepare_input_data(feature_names):
        features = {feature: 0 for feature in feature_names}

        features["Education"] = education_mapping[education]
        features["Years_Coding"] = years_coding
        country_feature = f"Country_{country}"
        if country_feature in features:
            features[country_feature] = 1

        for lang, selected in coding_skills.items():
            key = f"Codes_In_{lang}"
            if key in features:
                features[key] = int(selected)

        for feature, value in additional_feature_inputs.items():
            if feature in features:
                features[feature] = value

        input_df = pd.DataFrame([features])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Extra Debugging
        missing_cols = set(feature_names) - set(input_df.columns)
        extra_cols = set(input_df.columns) - set(feature_names)
        if missing_cols:
            st.warning(f"⚠️ Missing features: {sorted(missing_cols)}")
        if extra_cols:
            st.warning(f"⚠️ Extra features: {sorted(extra_cols)}")

        return input_df

    # --- 6. Prepare Input DataFrame ---
    input_df = prepare_input_data(all_feature_names)

    # --- 7. Show Input Data for Debugging ---
    if st.checkbox("Show Input Data for Debugging"):
        st.subheader("Input Data for Prediction")
        st.dataframe(input_df)
        st.text(f"Input Data Shape: {input_df.shape}")
        st.text(f"Input Features Count: {input_df.shape[1]}")
        st.text(f"Required Feature Count: 81")

        if input_df.shape[1] != 81:
            st.error(f"ERROR: Feature count mismatch! Input has {input_df.shape[1]}, model expects 81")

    # --- 8. Make Prediction ---
    if st.button("💰 Predict Salary"):
        try:
            if input_df.shape[1] != 81:
                st.error(f"ERROR: Input data has {input_df.shape[1]} features, but the model expects 81.")
            else:
                prediction = model.predict(input_df)[0]
                st.success(f"🎉 Estimated Annual Salary: ${prediction:,.2f}")
        except Exception as e:
            st.error(f"ERROR: An unexpected error occurred during prediction: {e}")
            import traceback
            st.error(traceback.format_exc())

    # --- 9. Footer ---
    st.markdown("---")
    st.markdown("Developed with ❤️ by Jiya, Rhea, and Michael", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
