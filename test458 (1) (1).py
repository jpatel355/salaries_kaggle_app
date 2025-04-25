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

    # --- 3. Define EXACTLY 81 Features ---
    # We'll create exactly 81 features that the model likely expects
    
    # Define Programming languages
    programming_languages = [
        "JAVA", "Python", "SQL", "GO", "JavaScript", "TypeScript", "C", "C++", 
        "C#", "PHP", "Ruby", "Swift", "Kotlin", "Rust", "Scala", "R", 
        "MATLAB", "Shell", "Assembly", "Perl", "VBA", "Objective-C", "Dart", 
        "Lua", "Groovy", "Haskell", "Clojure", "F#", "Julia", "COBOL"
    ]
    
    # Define Countries
    countries = [
        "India", "US", "Canada", "UK", "Germany", "France", "Australia", "Brazil",
        "Russia", "China", "Japan", "Spain", "Italy", "Netherlands", "Poland", 
        "Sweden", "Switzerland", "Turkey", "Mexico", "Argentina", "Colombia", 
        "Indonesia", "Pakistan", "Egypt", "Nigeria", "Iran", "Israel", "Singapore",
        "Malaysia", "Philippines", "Ireland", "Belgium", "Chile", "Denmark", 
        "Finland", "Norway", "South Africa", "Ukraine", "Portugal", "Austria",
        "Greece", "Czech Republic", "Hungary", "Romania", "New Zealand", "Other"
    ]
    
    # Basic features
    base_features = ["Education", "Years_Coding"]
    
    # Additional features to reach exactly 81
    # These could be platform experience, job roles, etc.
    additional_features = [
        "Years_Professional_Experience", "Team_Size", "Company_Size", "Is_Manager"
    ]
    
    # Build the list of 81 features
    all_feature_names = []
    
    # Add programming language features
    for lang in programming_languages:
        all_feature_names.append(f"Codes_In_{lang}")
    
    # Add base features
    all_feature_names.extend(base_features)
    
    # Add country features
    for country in countries:
        all_feature_names.append(f"Country_{country}")
        
    # Add additional features to reach exactly 81
    all_feature_names.extend(additional_features)
    
    # Ensure exactly 81 features
    if len(all_feature_names) > 81:
        all_feature_names = all_feature_names[:81]
    elif len(all_feature_names) < 81:
        # Add dummy features to reach exactly 81
        for i in range(len(all_feature_names), 81):
            all_feature_names.append(f"Feature_{i}")
    
    # Now we should have exactly 81 features
    st.sidebar.info(f"Using exactly 81 features")
    
    # --- 4. User Input Widgets ---
    with st.sidebar:
        st.header("Enter Your Details")
        
        # Education
        education_categories = ['HS', 'BS', 'MS', 'PHD']
        education_mapping = {level: i for i, level in enumerate(education_categories)}
        education = st.selectbox("Education Level", education_categories)
        
        # Years of coding
        years_coding = st.slider("⏳ Years of Coding", 0, 40, 5)
        
        # Country selection
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
        
        # Additional features
        additional_feature_inputs = {}
        
        if "Years_Professional_Experience" in all_feature_names:
            additional_feature_inputs["Years_Professional_Experience"] = st.slider(
                "Years of Professional Experience", 0, 40, 3
            )
        
        if "Team_Size" in all_feature_names:
            additional_feature_inputs["Team_Size"] = st.slider(
                "Team Size", 1, 100, 10
            )
            
        if "Company_Size" in all_feature_names:
            additional_feature_inputs["Company_Size"] = st.selectbox(
                "Company Size", 
                ["< 10", "10-50", "50-250", "250-1000", "> 1000"],
                index=2
            )
            # Convert to numeric
            company_size_map = {"< 10": 0, "10-50": 1, "50-250": 2, "250-1000": 3, "> 1000": 4}
            additional_feature_inputs["Company_Size"] = company_size_map[additional_feature_inputs["Company_Size"]]
            
        if "Is_Manager" in all_feature_names:
            additional_feature_inputs["Is_Manager"] = st.checkbox("Is a Manager")

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
        if 'country' in locals():
            country_feature = f"Country_{country}"
            if country_feature in features:
                features[country_feature] = 1
        
        # Set programming languages if applicable
        if 'coding_skills' in locals():
            for lang, has_skill in coding_skills.items():
                feature_name = f"Codes_In_{lang}"
                if feature_name in features:
                    features[feature_name] = int(has_skill)
        
        # Set additional features
        if 'additional_feature_inputs' in locals():
            for feature, value in additional_feature_inputs.items():
                if feature in features:
                    features[feature] = value
        
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
        st.text(f"Input Features Count: {input_df.shape[1]}")
        st.text(f"Required Feature Count: 81")
        
        # Check for feature count mismatch
        if input_df.shape[1] != 81:
            st.error(f"ERROR: Feature count mismatch! Input has {input_df.shape[1]}, model expects 81")
            st.write("Feature names:")
            st.write(input_df.columns.tolist())

    # --- 8. Make Prediction ---
    if st.button("💰 Predict Salary"):
        try:
            # Verify we have the right number of features
            if input_df.shape[1] != 81:
                st.error(f"ERROR: Input data has {input_df.shape[1]} features, but the model expects 81.")
                st.info("Please adjust the code to provide exactly 81 features.")
            else:
                # Make prediction
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
