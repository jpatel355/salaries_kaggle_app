import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

def main():
    st.title("📊 Salary Prediction App (Kaggle Survey 2022)")
    st.subheader("🔮 Predict your salary based on skills, experience, and education")

    education_mapping = {
        'HS': 0, 'BS': 1, 'MS': 2, 'PHD': 3, 'Associate': 1, 'Professional degree': 2,
        'I never completed any formal education': 0, 'Primary/elementary school': 0,
        'Some college/university study without earning a bachelor’s degree': 1,
        'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)': 0,
        'Some doctoral-level study without earning a doctorate': 3
    }
    education_options = list(education_mapping.keys())

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
        "Country_United States of America", "Country_Viet Nam"
    ]
    expected_features = len(feature_names)

    try:
        with open("gradient_boosting_pipeline.pkl", "rb") as f:
            model_dict = pickle.load(f)
            model = model_dict["model"]
            all_features_from_model = model_dict.get("columns")
            scaler = model_dict.get("scaler") or model_dict.get("standard_scaler")

        st.sidebar.success("✅ Model loaded successfully!")
        st.sidebar.write(f"🧠 Model type: {type(model).__name__}")
        if all_features_from_model:
            st.sidebar.write(f"✨ Features from model file: {len(all_features_from_model)}")
            st.sidebar.write(f"➡️ Using {expected_features} features for prediction")

        st.sidebar.header("⚙️ Input Features")
        codes_java = st.sidebar.checkbox("☕ I code in Java")
        codes_python = st.sidebar.checkbox("🐍 I code in Python")
        codes_sql = st.sidebar.checkbox("📊 I code in SQL")
        codes_go = st.sidebar.checkbox("🐹 I code in GO")
        years_coding = st.sidebar.slider("⏳ Years of Coding Experience", 0, 30, 5)
        education_str = st.sidebar.selectbox("🎓 Education Level", education_options)
        country = st.sidebar.selectbox("🌎 Country", [name.replace("Country_", "") for name in feature_names if name.startswith("Country_")])

        features = {
            "Codes_In_JAVA": int(codes_java),
            "Codes_In_Python": int(codes_python),
            "Codes_In_SQL": int(codes_sql),
            "Codes_In_GO": int(codes_go),
            "Years_Coding": years_coding,
            "Education": education_mapping.get(education_str, 0)
        }

        for country_name in [name.replace("Country_", "") for name in feature_names if name.startswith("Country_")]:
            features[f"Country_{country_name}"] = int(country == country_name)

        input_data = pd.DataFrame([features], columns=feature_names)

        if st.checkbox("🔍 Show input features"):
            st.write("Data:")
            st.write(input_data)

        if st.button("💰 Predict Salary"):
            try:
                if all_features_from_model is None:
                    st.error("🚨 Feature columns not found in model.")
                elif set(input_data.columns) != set(all_features_from_model):
                    st.error("🚨 Mismatch in feature columns.")
                    st.text(f"Difference:\n{set(all_features_from_model) ^ set(input_data.columns)}")
                else:
                    # Debugging: Print the column names from both the input and the model
                    st.text(f"Input Columns: {input_data.columns}")
                    st.text(f"Model Columns: {all_features_from_model}")

                    # Ensure all columns are present, filling with default values (e.g., 0)
                    missing_columns = set(all_features_from_model) - set(input_data.columns)
                    for col in missing_columns:
                        input_data[col] = 0

                    # Align columns
                    input_data = input_data[all_features_from_model]

                    if scaler:
                        scaled_array = scaler.transform(input_data)
                        input_data = pd.DataFrame(scaled_array, columns=all_features_from_model)

                    prediction = model.predict(input_data)
                    st.success(f"🎉 Estimated Annual Salary: ${prediction[0]:,.2f}")

                    if prediction[0] < 40000:
                        st.info("💼 Likely entry-level.")
                    elif prediction[0] < 80000:
                        st.info("🧑‍💻 Likely mid-level.")
                    elif prediction[0] < 120000:
                        st.info("👨‍💼 Likely senior-level.")
                    else:
                        st.info("🚀 Likely expert/leadership.")

            except Exception as e:
                st.error(f"🛑 Error during prediction: {e}")
                st.text("Model input info:")
                st.text(f"Expected shape: {model.n_features_in_ if hasattr(model, 'n_features_in_') else 'Unknown'}")
                st.text(f"Input shape: {input_data.shape}")

    except FileNotFoundError:
        st.error("File not found: 'gradient_boosting_pipeline.pkl' 📂")
    except Exception as e:
        st.error(f"Error loading model: {e} 🤕")

    st.markdown("---")
    st.markdown("<small>✨ Built with ❤️ using Streamlit — Jiya, Rhea, and Michael", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
