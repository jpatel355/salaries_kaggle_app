import streamlit as st
import pandas as pd
import pickle

def main():
    # Load the trained model first to ensure it's ready
    try:
        with open("salary2022_model.pkl", "rb") as f:
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

    # **COMPLETE LIST OF 42 FEATURES (REPLACE WITH YOUR ACTUAL FEATURES)**
    feature_names = [
        "Codes_In_JAVA", "Codes_In_Python", "Codes_In_SQL", "Codes_In_GO", "Years_Coding", "Education",
        "Country_India", "Country_US", "Country_Spain", "Country_Canada", "Country_Germany",
        "Country_France", "Country_Japan", "Country_UK", "Country_Australia", "Country_Brazil",
        "Country_Russia", "Country_China", "Country_South Africa", "Country_Netherlands",
        "Country_Poland", "Country_Italy", "Country_Sweden", "Country_Switzerland", "Country_Turkey",
        "Country_Mexico", "Country_Argentina", "Country_Colombia", "Country_Indonesia",
        "Country_Pakistan", "Country_Egypt", "Country_Nigeria", "Country_Iran", "Country_Israel",
        "Country_Singapore", "Country_Malaysia", "Country_Philippines", "Country_Ireland",
        "Country_Belgium", "Country_Chile", "Country_Denmark", "Country_Finland", "Country_Norway",
        "Country_Other" # Catch-all for less frequent countries
        # Add any other features if your model used them
    ]

    # Sidebar inputs
    education = st.sidebar.selectbox("Education Level", education_options)
    years_coding = st.sidebar.slider("⏳ Years of Coding Experience", 0, 40, 5)
    country = st.sidebar.selectbox("🌎 Country", ["India", "US", "Canada", "Spain", "Germany", "France", "Japan", "UK", "Australia", "Brazil", "Russia", "China", "South Africa", "Netherlands", "Poland", "Italy", "Sweden", "Switzerland", "Turkey", "Mexico", "Argentina", "Colombia", "Indonesia", "Pakistan", "Egypt", "Nigeria", "Iran", "Israel", "Singapore", "Malaysia", "Philippines", "Ireland", "Belgium", "Chile", "Denmark", "Finland", "Norway", "Other"])
    codes_java = st.sidebar.checkbox("☕ I code in Java")
    codes_python = st.sidebar.checkbox("🐍 I code in Python")
    codes_sql = st.sidebar.checkbox("📊 I code in SQL")
    codes_go = st.sidebar.checkbox("🐹 I code in GO")

    education_num = education_mapping[education]

    # Initialize the feature dictionary with all expected features set to 0
    features = {feature: 0 for feature in feature_names}

    # Update the features based on user input
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
    elif "Country_Other" in features:
        features["Country_Other"] = 1

    # Create input DataFrame with all expected columns
    input_data = pd.DataFrame([features], columns=feature_names)

    if st.checkbox("🔍 Show input features"):
        st.write(input_data)

    if st.button("💰 Predict Salary"):
        try:
            # Ensure the input has all the features the model expects
            if input_data.shape[1] == len(feature_names):
                prediction = model.predict(input_data)[0]
                st.success(f"🎉 Estimated Annual Salary: ${prediction:,.2f}")
            else:
                st.error(f"Error: Input data has {input_data.shape[1]} features, but the model expects {len(feature_names)}.")
        except Exception as e:
            st.error(f"Error making prediction: {e} 🤕")

    st.markdown("---")
    st.markdown("<small>✨ Built with ❤️ using Streamlit — Jiya, Rhea, and Michael", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
