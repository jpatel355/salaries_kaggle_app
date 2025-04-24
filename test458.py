import streamlit as st
import pandas as pd
import pickle
import numpy as np

def main():
    st.title("Salary Prediction App (Kaggle Survey 2022)")

    # Load the pickled dictionary
    try:
        with open("Salary2022_model.pkl", "rb") as f:
            model_dict = pickle.load(f)

        # Extract the model and feature names
        model = model_dict["model"]
        all_feature_names = model_dict["columns"]
        expected_features = 42
        feature_names = all_feature_names[:expected_features]

        st.sidebar.success("Model loaded successfully!")
        st.sidebar.write(f"Model type: {type(model).__name__}")
        st.sidebar.write(f"Available features in model file: {len(all_feature_names)}")
        st.sidebar.write(f"Using features for prediction: {len(feature_names)}")

        if len(all_feature_names) > expected_features:
            removed_features = all_feature_names[expected_features:]
            st.sidebar.warning(f"Not using these extra features: {removed_features}")

    except FileNotFoundError:
        st.error("Error: Model file 'Salary2022_model.pkl' not found.")
        return
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Input Widgets
    st.sidebar.header("Input Features")
    st.sidebar.subheader("Programming Languages")
    codes_java = st.sidebar.checkbox("I code in Java")
    codes_python = st.sidebar.checkbox("I code in Python")
    codes_sql = st.sidebar.checkbox("I code in SQL", value=True)
    codes_go = st.sidebar.checkbox("I code in GO")
    years_coding = st.sidebar.slider("Years of Coding Experience", 0, 30, 5)
    education = st.sidebar.selectbox("Education Level", [0, 1, 2, 3, 4, 5])
    country = st.sidebar.selectbox("Country", ["India", "US", "Other"])

    feature_dict = {}
    feature_dict["Codes_In_JAVA"] = int(codes_java)
    feature_dict["Codes_In_Python"] = int(codes_python)
    feature_dict["Codes_In_SQL"] = int(codes_sql)
    feature_dict["Codes_In_GO"] = int(codes_go)
    feature_dict["Years_Coding"] = years_coding
    feature_dict["Education"] = education

    if "Country_India" in feature_names:
        feature_dict["Country_India"] = 1 if country == "India" else 0
    if "Country_US" in feature_names:
        feature_dict["Country_US"] = 1 if country == "US" else 0
    if "Country_Other" in feature_names:
        feature_dict["Country_Other"] = 1 if country == "Other" else 0

    for feature in feature_names:
        if feature not in feature_dict:
            feature_dict[feature] = 0

    input_data = pd.DataFrame([{name: feature_dict.get(name, 0) for name in feature_names}])

    if st.checkbox("Show input features"):
        st.write("Input Data:")
        st.write(input_data)
        st.info(f"Total features in input: {input_data.shape[1]}")

    if st.button("Predict Salary"):
        try:
            if input_data.shape[1] != expected_features:
                st.error(f"Error: Expected {expected_features} features, got {input_data.shape[1]}.")
            else:
                # *** ADD THESE LINES FOR DEBUGGING ***
                st.write("Model Feature Names (first 10):", feature_names[:10])
                st.write("Input Data Column Names:", input_data.columns.tolist()[:10])
                # *************************************

                prediction = model.predict(input_data)
                st.success(f"Predicted Annual Salary: ${prediction[0]:,.2f}")

                if prediction[0] < 40000:
                    st.info("Likely entry-level.")
                elif prediction[0] < 80000:
                    st.info("Likely mid-level.")
                elif prediction[0] < 120000:
                    st.info("Likely senior-level.")
                else:
                    st.info("Likely expert/leadership.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.write("Input data shape:", input_data.shape)
            if hasattr(model, 'n_features_in_'):
                st.write("Model expected features:", model.n_features_in_)
            st.info("Check error and ensure input matches model expectations.")

    if st.sidebar.checkbox("Debug Mode"):
        st.subheader("Debug Information")

        if st.button("Inspect Model"):
            try:
                if hasattr(model, 'n_features_in_'):
                    st.write("Model expected feature count:", model.n_features_in_)
                if hasattr(model, 'feature_names_in_'):
                    st.write("Model expected feature names:", model.feature_names_in_)
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    st.write("Top 10 important features:")
                    for i in range(min(10, len(indices))):
                        st.write(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
            except Exception as debug_error:
                st.error(f"Debug error: {debug_error}")

if __name__ == "__main__":
    main()
