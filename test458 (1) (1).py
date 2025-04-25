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

    # --- 2. Set up UI Elements ---
    st.title("💼 Salary Predictor")
    st.subheader("Accurate salary predictions based on your profile.")

    # --- 3. Extract Model Features ---
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_.tolist()
        num_model_features = len(model_features)
        st.sidebar.info(f"Model expects {num_model_features} features")
    else:
        st.error("Cannot extract feature names from model. Please check model type.")
        return

    # --- 4. Display Model Info for Debugging ---
    if st.checkbox("Show Model Information"):
        st.write("Model type:", type(model))
        st.write("Number of features expected:", num_model_features)
        st.write("Model features:")
        
        # Group features by type for better display
        feature_groups = {
            "Programming Languages": [],
            "Countries": [],
            "Education/Experience": [],
            "Other": []
        }
        
        for feature in model_features:
            if feature.startswith("Codes_In_"):
                feature_groups["Programming Languages"].append(feature)
            elif feature.startswith("Country_"):
                feature_groups["Countries"].append(feature)
            elif feature in ["Education", "Years_Coding"]:
                feature_groups["Education/Experience"].append(feature)
            else:
                feature_groups["Other"].append(feature)
        
        for group, features in feature_groups.items():
            st.write(f"**{group} ({len(features)}):**")
            st.write(", ".join(features))

    # --- 5. Extract Categories from Model Features ---
    # Extract programming languages
    programming_languages = [lang.replace("Codes_In_", "") for lang in model_features if lang.startswith("Codes_In_")]
    
    # Extract countries
    countries = [country.replace("Country_", "") for country in model_features if country.startswith("Country_")]
    
    # Check for education feature
    has_education = "Education" in model_features
    education_categories = ['HS', 'BS', 'MS', 'PHD']
    education_mapping = {level: i for i, level in enumerate(education_categories)}
    
    # Check for years coding feature
    has_years_coding = "Years_Coding" in model_features
    
    # Identify other features not covered by our basic categories
    other_features = [f for f in model_features if not (f.startswith("Codes_In_") or f.startswith("Country_") or f in ["Education", "Years_Coding"])]
    
    # --- 6. User Input Widgets ---
    with st.sidebar:
        st.header("Enter Your Details")
        
        # Education
        if has_education:
            education = st.selectbox("Education Level", education_categories)
        
        # Years of coding
        if has_years_coding:
            years_coding = st.slider("⏳ Years of Coding", 0, 40, 5)
        
        # Country selection
        if countries:
            country = st.selectbox("🌎 Country", countries)
        
        # Programming languages
        if programming_languages:
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
            for i, lang in enumerate(programming_languages):
                emoji = emoji_map.get(lang, "💻")
                if i % 2 == 0:
                    coding_skills[lang] = col1.checkbox(f"{emoji} {lang}")
                else:
                    coding_skills[lang] = col2.checkbox(f"{emoji} {lang}")
        
        # Handle other features
        if other_features:
            st.subheader("Other Factors")
            other_values = {}
            for feature in other_features:
                # Make a reasonable guess about the feature type and provide appropriate input
                if feature.startswith("Has_") or feature.startswith("Is_"):
                    other_values[feature] = st.checkbox(f"{feature.replace('_', ' ')}")
                elif "Years" in feature or "Age" in feature:
                    other_values[feature] = st.slider(f"{feature.replace('_', ' ')}", 0, 50, 5)
                else:
                    other_values[feature] = st.number_input(f"{feature.replace('_', ' ')}", 0, 100, 0)

    # --- 7. Prepare Input Data Function ---
    def prepare_input_data(model_features):
        """
        Creates a feature vector with exactly the features the model expects
        """
        features = {feature: 0 for feature in model_features}
        
        # Set education if applicable
        if has_education and "Education" in features:
            features["Education"] = education_mapping[education]
        
        # Set years coding if applicable
        if has_years_coding and "Years_Coding" in features:
            features["Years_Coding"] = years_coding
        
        # Set country if applicable
        if countries:
            country_feature = f"Country_{country}"
            if country_feature in features:
                features[country_feature] = 1
        
        # Set programming languages if applicable
        if programming_languages:
            for lang, has_skill in coding_skills.items():
                feature_name = f"Codes_In_{lang}"
                if feature_name in features:
                    features[feature_name] = int(has_skill)
        
        # Set other features if applicable
        if other_features:
            for feature, value in other_values.items():
                if feature in features:
                    features[feature] = value
        
        # Create DataFrame with features in exact order expected by model
        input_df = pd.DataFrame([features], columns=model_features)
        return input_df

    # --- 8. Prepare Input DataFrame ---
    input_df = prepare_input_data(model_features)
    
    # --- 9. Show Input Data for Debugging ---
    if st.checkbox("Show Input Data for Debugging"):
        st.subheader("Input Data for Prediction")
        st.dataframe(input_df)
        st.text(f"Input Data Shape: {input_df.shape}")
        st.text(f"Input Data Features Count: {input_df.shape[1]}")
        st.text(f"Model Expected Features Count: {num_model_features}")
        
        # Check for feature count mismatch
        if input_df.shape[1] != num_model_features:
            st.error(f"ERROR: Feature count mismatch! Input has {input_df.shape[1]}, model expects {num_model_features}")
            
            # Compare features to find missing or extra
            input_features = set(input_df.columns)
            model_features_set = set(model_features)
            
            missing = model_features_set - input_features
            extra = input_features - model_features_set
            
            if missing:
                st.error(f"Features missing from input data: {missing}")
            if extra:
                st.error(f"Extra features in input data: {extra}")

    # --- 10. Make Prediction ---
    if st.button("💰 Predict Salary"):
        try:
            # Verify we have the right number of features
            if input_df.shape[1] != num_model_features:
                st.error(f"ERROR: Input data has {input_df.shape[1]} features, but the model expects {num_model_features}.")
                st.info("Use the 'Show Model Information' and 'Show Input Data for Debugging' options to identify missing features.")
            else:
                # Ensure features are in the exact order model expects
                input_df = input_df[model_features]
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                st.success(f"🎉 Estimated Annual Salary: ${prediction:,.2f}")
                
                # Show feature importance if available
                if hasattr(model, 'feature_importances_'):
                    if st.checkbox("Show Feature Importance"):
                        importance = model.feature_importances_
                        feature_imp = pd.DataFrame({'Feature': model_features, 'Importance': importance})
                        feature_imp = feature_imp.sort_values('Importance', ascending=False).head(10)
                        
                        st.subheader("Top 10 Features Affecting Salary")
                        st.bar_chart(feature_imp.set_index('Feature'))
        except Exception as e:
            st.error(f"ERROR: An unexpected error occurred during prediction: {e}")
            st.info("Inspect the model information and input data to identify the issue.")

    # --- 11. Footer ---
    st.markdown("---")
    st.markdown("Developed with ❤️ by Jiya, Rhea, and Michael", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
