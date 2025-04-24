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
        st.error("File not found: 'salary2022_model.pkl' 📂")
        return
    except Exception as e:
        st.error(f"Error loading model: {e} 🤕")
        return

    st.title("💼 Salary Predictor")
    st.subheader("📈 Predict your salary based on skills, experience, and education")

    education_mapping = {'HS': 0, 'BS': 1, 'MS': 2, 'PHD': 3}
    education_options = list(education_mapping.keys())

    # Full list of features the model expects (this should match the training dataset's feature list)
    feature_names = [
        "Codes_In_JAVA", "Codes_In_Python", "Codes_In_SQL", "Codes_In_GO", "Years_Coding", "Education",
        "Country_India", "Country_US", "Country_Spain", "Country_Other",
        # Add any other features that the model expects here, such as:
        # "Country_Canada", "Country_Germany", "Country_China", etc.
    ]

    education = st.sidebar.selectbox("Education Level", education_options)
    years_coding = st.sidebar.slider("⏳ Years of Coding Experience", 0, 40, 5)
    country = st.sidebar.selectbox("🌎 Country", ["India", "US", "Canada", "Spain", "Other"])
    codes_java = st.sidebar.checkbox("☕ I code in Java")
    codes_python = st.sidebar.checkbox("🐍 I code in Python")
    codes_sql = st.sidebar.checkbox("📊 I code in SQL")
    codes_go = st.sidebar.checkbox("🐹 I code in GO")

    education_num = education_mapping[education]

    # Creating the input features
    features = {
        "Education": education_num,
        "Years_Coding": years_coding,
        "Codes_In_JAVA": int(codes_java),
        "Codes_In_Python": int(codes_python),
        "Codes_In_SQL": int(codes_sql),
        "Codes_In_GO": int(codes_go),
        # Initializing country variables to 0
        "Country_India": 0,
        "Country_Other": 0,
        "Country_Spain": 0,
        "Country_US": 0,
        # Add additional country variables here if necessary
    }

    # One-hot encoding for country
    if country != "Canada":
        if country == "India":
            features["Country_India"] = 1
        elif country == "US":
            features["Country_US"] = 1
        elif country == "Spain":
            features["Country_Spain"] = 1
        elif country == "Other":
            features["Country_Other"] = 1

    # Ensure that the number of features matches the model's expectations
    input_data = pd.DataFrame([features], columns=feature_names)

    if st.checkbox("🔍 Show input features"):
        st.write(input_data)

    if st.button("💰 Predict Salary"):
        try:
            # Ensure the model is able to predict
            prediction = model.predict(input_data)[0]
            st.success(f"🎉 Estimated Annual Salary: ${prediction:,.2f}")
        except Exception as e:
            st.error(f"Error making prediction: {e} 🤕")

    st.markdown("---")
    st.markdown("<small>✨ Built with ❤️ using Streamlit — Jiya, Rhea, and Michael", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
