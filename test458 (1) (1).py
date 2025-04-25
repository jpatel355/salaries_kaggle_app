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
        
        # Display model information
        if st.checkbox("Show Model Information"):
            st.write("Model type:", type(model))
            if hasattr(model, 'n_features_in_'):
                st.write("Number of features expected:", model.n_features_in_)
            else:
                st.write("Could not determine number of features expected")
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_filename}' not found. Please ensure it's in the correct location. 📂")
        return
    except Exception as e:
        st.error(f"Error loading model: {e} 🤕")
        return

    # --- 2. Set up UI Elements ---
    st.title("💼 Salary Predictor")
    st.subheader("Accurate salary predictions based on your profile.")

    # --- 3. Define Expected Features ---
    # Since we can't extract feature names, we'll define a comprehensive list of features
    # that covers what the model likely expects
    
    # Programming languages that are commonly used
    programming_languages = [
        "JAVA", "Python", "SQL", "GO", "JavaScript", "TypeScript", "C", "C++", 
        "C#", "PHP", "Ruby", "Swift", "Kotlin", "Rust", "Scala", "R", 
        "MATLAB", "Shell", "Assembly", "Perl", "VBA", "Objective-C", "Dart", 
        "Lua", "Groovy", "Haskell", "Clojure", "F#", "Julia", "COBOL"
    ]
    
    # Countries commonly found in developer surveys
    countries = [
        "India", "US", "Canada", "UK", "Germany", "France", "Australia", "Brazil",
        "Russia", "China", "Japan", "Spain", "Italy", "Netherlands", "Poland", 
        "Sweden", "Switzerland", "Turkey", "Mexico", "Argentina", "Colombia", 
        "Indonesia", "Pakistan", "Egypt", "Nigeria", "Iran", "Israel", "Singapore",
        "Malaysia", "Philippines", "Ireland", "Belgium", "Chile", "Denmark", 
        "Finland", "Norway", "South Africa", "Ukraine", "Portugal", "Austria",
        "Greece", "Czech Republic", "Hungary", "Romania", "New Zealand", "Other"
    ]
    
    # Base features for education and experience
    base_features = ["Education", "Years_Coding"]
    
    # Combine all potential features
    all_feature_names = []
    
    # Add programming language features
    for lang in programming_languages:
        all_feature_names.append(f"Codes_In_{lang}")
    
    # Add base features
    all_feature_names.extend(base_features)
    
    # Add country features
    for country in countries:
        all_feature_names.append(f"Country_{country}")

    # Get expected number of features from model if available
    num_expected_features = 81  # Default to 81 as mentioned in the error
    if hasattr(model, 'n_features_in_'):
        num_expected_features = model.n_features_in_
    
    st.sidebar.info(f"Model expects {num_expected_features} features")
    
    # Limit the feature list to exactly what the model expects
    # If we have more features than needed, prioritize:
    # 1. Basic features (Education, Years_Coding)
    # 2. Major programming languages
    # 3. Major countries
    if len(all_feature_names) > num_expected_features:
        all_feature_names = all_feature_names[:num_expected_features]
    
    # --- 4. User Input Widgets ---
    with st.sidebar:
        st.header("Enter Your Details")
        
        # Education
        education_categories = ['HS', 'BS', 'MS', 'PHD']
        education_mapping = {level: i for i, level in enumerate(education_categories)}
        education = st.selectbox("Education Level", education_categories)
        
        # Years of coding
        years_coding = st.slider("⏳ Years of Coding", 0, 40, 5)
        
        # Country selection - limit to the ones in our feature set
        country_features = [f for f in all_feature_names if f.startswith("Country_")]
        available_countries = [c.replace("Country_", "") for c in country_features]
        
        if available_countries:
            country = st.selectbox("🌎 Country", available_countries)
        
        # Programming languages
        lang_features = [f for f in all_feature_names if f.startswith("Codes_In_")]
        available_languages = [l.replace("Codes_In_", "") for l in lang_features]
        
        if available_languages:
            st.subheader("Programming Languages")
            coding_skills = {}
            
            # Create emoji mapping for common languages
            emoji_map = {
                "JAVA": "☕", "Python": "🐍", "SQL": "📊", "GO": "🐹", 
                "JavaScript": "📜", "TypeScript": "🔷", "C": "🔣", "C++": "🧮", 
                "C#": "🎮", "PHP": "🐘", "Ruby": "💎", "Swift": "🍎", 
                "Kotlin": "🤖", "Rust": "⚙️", "Scala": "📈", "R": "📊", 
                "MATLAB": "🧮", "Shell": "🐚", "Assembly": "⚡", "Perl": "🐪"
            }
            
            # Create checkboxes for languages, organized in columns for better UI
            col1, col2 = st.columns(2)
            for i, lang in enumerate(available_languages):
                emoji = emoji_map.get(lang, "💻")
                if i % 2 == 0:
                    coding_skills[lang] = col1.checkbox(f"{emoji} {lang}")
                else:
                    coding_skills[lang] = col2.checkbox(f"{emoji} {lang}")

    # --- 5. Prepare Input Data Function ---
    def prepare_input_data(feature_names):
        """
        Creates a feature vector with exactly the features the model expects
        """
        features = {feature: 0 for feature in feature_names}
        
        # Set education if applicable
        if "Education" in features:
            features["Education"] = education_mapping[education]
        
        # Set years coding if applicable
        if "Years_Coding" in features:
            features["Years_Coding"] = years_coding
        
        # Set country if applicable
        country_feature = f"Country_{country}"
        if country_feature in features:
            features[country_feature] = 1
        
        # Set programming languages if applicable
        for lang, has_skill in coding_skills.items():
            feature_name = f"Codes_In_{lang}"
            if feature_name in features:
                features[feature_name] = int(has_skill)
        
        # Create DataFrame with features
        input_df = pd.DataFrame([features])
        return input_df

    # --- 6. Prepare Input DataFrame ---
    input_df = prepare_input_data(all_feature_names)
    
    # --- 7. Show Input Data for Debugging ---
    if st.checkbox("Show Input Data for Debugging"):
        st.subheader("Input Data for Prediction")
        st.dataframe(input_df)
        st.text(f"Input Data Shape: {input_df.shape}")
        st.text(f"Input Data Features Count: {input_df.shape[1]}")
        st.text(f"Model Expected Features Count: {num_expected_features}")
        
        # Check for feature count mismatch
        if input_df.shape[1] != num_expected_features:
            st.error(f"ERROR: Feature count mismatch! Input has {input_df.shape[1]}, model expects {num_expected_features}")

    # --- 8. Make Prediction ---
    if st.button("💰 Predict Salary"):
        try:
            # Verify we have the right number of features
            if input_df.shape[1] != num_expected_features:
                st.error(f"ERROR: Input data has {input_df.shape[1]} features, but the model expects {num_expected_features}.")
                st.info("Please adjust the code to provide exactly the number of features the model expects.")
            else:
                # Make prediction
                prediction = model.predict(input_df)[0]
                st.success(f"🎉 Estimated Annual Salary: ${prediction:,.2f}")
        except Exception as e:
            st.error(f"ERROR: An unexpected error occurred during prediction: {e}")
            st.info("Try to determine what features the model was trained with.")

    # --- 9. Footer ---
    st.markdown("---")
    st.markdown("Developed with ❤️ by Jiya, Rhea, and Michael", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
