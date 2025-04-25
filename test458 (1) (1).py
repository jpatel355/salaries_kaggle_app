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
        
        if st.checkbox("Show Model Information"):
            st.write("Model type:", type(model))
            if hasattr(model, 'n_features_in_'):
                st.write("Number of features expected:", model.n_features_in_)
            else:
                st.write("Could not determine number of features expected")
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_filename}' not found.")
        return
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # --- 2. UI Elements ---
    st.title("💼 Salary Predictor")
    st.subheader("Accurate salary predictions based on your profile.")

    # --- 3. Feature Definitions (81 total) ---
    programming_languages = [
        "JAVA", "Python", "SQL", "GO", "JavaScript", "TypeScript", "C", "C++", 
        "C#", "PHP", "Ruby", "Swift", "Kotlin", "Rust", "Scala", "R", 
        "MATLAB", "Shell", "Assembly", "Perl", "VBA", "Objective-C", "Dart", 
        "Lua", "Groovy", "Haskell", "Clojure", "F#", "Julia", "COBOL"
    ]
    
    countries = [
        "India", "US", "Canada", "UK", "Germany", "France", "Australia", "Brazil",
        "Russia", "China", "Japan", "Spain", "Italy", "Netherlands", "Poland", 
        "Sweden", "Switzerland", "Turkey", "Mexico", "Argentina", "Colombia", 
        "Indonesia", "Pakistan", "Egypt", "Nigeria", "Iran", "Israel", "Singapore",
        "Malaysia", "Philippines", "Ireland", "Belgium", "Chile", "Denmark", 
        "Finland", "Norway", "South Africa", "Ukraine", "Portugal", "Austria",
        "Greece", "Czech Republic", "Hungary", "Romania", "New Zealand", "Other"
    ]
    
    base_features = ["Education", "Years_Coding"]
    additional_features = ["Years_Professional_Experience", "Team_Size", "Company_Size", "Is_Manager"]
    
    all_feature_names = (
        [f"Codes_In_{lang}" for lang in programming_languages] +
        base_features +
        [f"Country_{c}" for c in countries] +
        additional_features
    )
    
    # Ensure exactly 81 features
    if len(all_feature_names) < 81:
        all_feature_names += [f"Feature_{i}" for i in range(len(all_feature_names), 81)]
    elif len(all_feature_names) > 81:
        all_feature_names = all_feature_names[:81]
    
    st.sidebar.info("Using exactly 81 features")

    # --- 4. Sidebar Input ---
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
        col1, col2 = st.columns(2)
        emoji_map = {
            "JAVA": "☕", "Python": "🐍", "SQL": "📊", "GO": "🐹", 
            "JavaScript": "📜", "TypeScript": "🔷", "C": "🔣", "C++": "🧮", 
            "C#": "🎮", "PHP": "🐘", "Ruby": "💎", "Swift": "🍎", 
            "Kotlin": "🤖", "Rust": "⚙️", "Scala": "📈", "R": "📊", 
            "MATLAB": "🧮", "Shell": "🐚", "Assembly": "⚡", "Perl": "🐪"
        }
        for i, lang in enumerate(available_languages):
            emoji = emoji_map.get(lang, "💻")
            if i % 2 == 0:
                coding_skills[lang] = col1.checkbox(f"{emoji} {lang}")
            else:
                coding_skills[lang] = col2.checkbox(f"{emoji} {lang}")
        
        additional_feature_inputs = {}
        additional_feature_inputs["Years_Professional_Experience"] = st.slider(
            "Years of Professional Experience", 0, 40, 3
        )
        additional_feature_inputs["Team_Size"] = st.slider("Team Size", 1, 100, 10)
        company_size_map = {"< 10": 0, "10-50": 1, "50-250": 2, "250-1000": 3, "> 1000": 4}
        cs = st.selectbox("Company Size", list(company_size_map.keys()), index=2)
        additional_feature_inputs["Company_Size"] = company_size_map[cs]
        additional_feature_inputs["Is_Manager"] = st.checkbox("Is a Manager")

    # --- 5. Updated prepare_input_data() ---
    def prepare_input_data(feature_names):
        features = {feature: 0 for feature in feature_names}
        
        if "Education" in features:
            features["Education"] = education_mapping.get(education, 0)
        
        if "Years_Coding" in features:
            features["Years_Coding"] = years_coding
        
        country_feature = f"Country_{country}"
        if country_feature in features:
            features[country_feature] = 1
        
        for lang, has_skill in coding_skills.items():
            feature_name = f"Codes_In_{lang}"
            if feature_name in features:
                features[feature_name] = int(has_skill)
        
        for feature, value in additional_feature_inputs.items():
            if feature in features:
                features[feature] = value

        input_df = pd.DataFrame([features])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        return input_df

    # --- 6. Prepare DataFrame ---
    input_df = prepare_input_data(all_feature_names)

    # --- 7. Debug Info ---
    if st.checkbox("Show Input Data for Debugging"):
        st.subheader("Input Data for Prediction")
        st.dataframe(input_df)
        st.text(f"Input Data Shape: {input_df.shape}")
        if input_df.shape[1] != 81:
            st.error("Feature count mismatch! Please review your feature list.")

    # --- 8. Prediction ---
    if st.button("💰 Predict Salary"):
        try:
            if input_df.shape[1] != 81:
                st.error("Feature count mismatch! Model expects exactly 81 features.")
            else:
                prediction = model.predict(input_df)[0]
                st.success(f"🎉 Estimated Annual Salary: ${prediction:,.2f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
            import traceback
            st.error(traceback.format_exc())

    # --- 9. Footer ---
    st.markdown("---")
    st.markdown("Developed with ❤️ by Jiya, Rhea, and Michael")

if __name__ == "__main__":
    main()
