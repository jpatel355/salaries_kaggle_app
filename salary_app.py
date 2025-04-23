import streamlit as st
import pickle
import os

def main():
    st.title("💰 Salary Prediction App 💰")

    # Provide the correct path to the model file
    model_file_path = os.path.join('.', 'kaggle2022_model (3).pkl')  # Ensure the model file is in the same directory

    # Load the pickled model and column list
    try:
        with open(model_file_path, 'rb') as file:
            model_bundle = pickle.load(file)
            model = model_bundle['model']
            model_columns = model_bundle['columns']
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please ensure the file '{model_file_path}' is in the same directory.")
        return  
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Rest of your app code follows...
    # Ensure your CSV file is in the correct directory or provide the correct path
    csv_file_path = os.path.join('.', 'kaggle_survey_2022_responses.csv')
    if not os.path.exists(csv_file_path):
        st.error(f"Error: CSV file not found at {csv_file_path}.")
        return

    # Continue with the rest of your app logic...
    # Your input handling and prediction logic follows...

if __name__ == '__main__':
    main()
