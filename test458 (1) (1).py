import streamlit as st
import pandas as pd
import pickle

def main():
    # Load the trained model first to ensure it's ready
    try:
        with open("salary2022_model (2).pkl", "rb") as f:
            model = pickle.load(f)
        st.sidebar.success("✅ Model loaded successfully!")
    except FileNotFoundError:
        st.error("File not found: 'salary2022_model (2).pkl' 📂")
        return
    except Exception as e:
        st.error(f"Error loading model: {e} 🤕")
        return

    st.title("💼 Salary Predictor")
    st.subheader("📈 Predict your salary based on skills, experience, and education")

    education_mapping = {'HS': 0, 'BS': 1, 'MS': 2, 'PHD': 3}
    education_options = list(education_mapping.keys())

    # **EXACT LIST OF 42 FEATURES FROM YOUR TRAINING DATA (ENSURE THIS IS CORRECT)**
    expected_feature_names = [
        "Codes_In_JAVA", "Codes_In_Python", "Codes_In_SQL", "Codes_In_GO", "Years_Coding", "Education",
        "Country_India", "Country_US", "Country_Spain", "Country_Canada", "Country_Germany",
        "Country_France", "Country_Japan", "Country_UK", "Country_Australia", "Country_Brazil",
        "Country_Russia", "Country_China", "Country_South Africa", "Country_Netherlands",
        "Country_Poland", "Country_Italy", "Country_Sweden", "Country_Switzerland", "Country_Turkey",
        "Country_Mexico", "Country_Argentina", "Country_Colombia", "Country_Indonesia",
        "Country_Pakistan", "Country_Egypt", "Country_Nigeria", "Country_Iran", "Country_Israel",
        "Country_Singapore", "Country_Malaysia", "Country_Philippines", "Country_Ireland",
        "Country_Belgium", "Country_Chile", "Country_Denmark", "Country_Finland" # Example - YOU NEED THE FULL 42
    ]
    num_expected_features = len(expected_feature_names)

    # List of all possible countries in the dropdown
    all_countries = ["India", "US", "Canada", "Spain", "Germany", "France", "Japan", "UK", "Australia", "Brazil", "Russia", "China", "South Africa", "Netherlands", "Poland", "Italy", "Sweden", "Switzerland", "Turkey", "Mexico", "Argentina", "Colombia", "Indonesia", "Pakistan", "Egypt", "Nigeria", "Iran", "Israel", "Singapore", "Malaysia", "Philippines", "Ireland", "Belgium", "Chile", "Denmark", "Finland", "Norway", "Other"]

    # Sidebar inputs
    education = st.sidebar.selectbox("Education Level", education_options)
    years_coding = st.sidebar.slider("⏳ Years of Coding Experience", 0, 40, 5)
    country = st.sidebar.selectbox("🌎 Country", all_countries)
    codes_java = st.sidebar.checkbox("☕ I code in Java")
    codes_python = st.sidebar.checkbox("🐍 I code in Python")
    codes_sql = st.sidebar.checkbox("📊 I code in SQL")
    codes_go = st.sidebar.checkbox("🐹 I code in GO")

    education_num = education_mapping[education]

    # Initialize the feature dictionary with all expected features set to 0
    features = {feature: 0 for feature in expected_feature_names}

    # Update the base numerical/boolean features
    features["Education"] = education_num
    features["Years_Coding"] = years_coding
    features["Codes_In_JAVA"] = int(codes_java)
    features["Codes_In_Python"] = int(codes_python)
    features["Codes_In_SQL"] = int(codes_sql)
    features["Codes_In_GO"] = int(codes_go)

    # One-hot encode the selected country
    country_column = f"Country_{country}"
    if country_column in features:
        features[country_column] = 1
    elif "Country_Other" in features and country not in [c.split("_")[1] if "_" in c and c.startswith("Country_") else "" for c in expected_feature_names]:
        features["Country_Other"] = 1

    # Create input DataFrame with the EXACTLY expected columns
    input_data = pd.DataFrame([features], columns=expected_feature_names)

    if st.checkbox("🔍 Show input features"):
        st.write("Input Data:")
        st.write(input_data)
        st.info(f"Number of features in input data: {input_data.shape[1]}")
        st.info(f"Number of expected features: {num_expected_features}")
        st.info(f"Columns in input data: {input_data.columns.tolist()}")
        st.info(f"Expected columns: {expected_feature_names}")

    if st.button("💰 Predict Salary"):
        try:
            # Ensure the input has the correct number of features
            if input_data.shape[1] == num_expected_features:
                prediction = model.predict(input_data)[0]
                st.success(f"🎉 Estimated Annual Salary: ${prediction:,.2f}")
            else:
                st.error(f"Error: Input data has {input_data.shape[1]} features, but the model expects {num_expected_features}.")
        except Exception as e:
            st.error(f"Error making prediction: {e} 🤕")

    st.markdown("---")
    st.markdown("<small>✨ Built with ❤️ using Streamlit — Jiya, Rhea, and Michael", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
