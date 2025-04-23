import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

def main():
    st.title("💰 Salary Prediction App 💰")  # Added a title with emojis

    # Load the pickled model
    with open('salary_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # Create input widgets with a more organized layout
    st.sidebar.header("Input Features ⚙️")  # Added a sidebar header

    # Use the pre-processed data to get country list.
    age = st.sidebar.slider("Age 🎂", 18, 65, 30)
    country = st.sidebar.selectbox("Country 🌍", X_train['Country'].unique()) # Change to X_train
    education = st.sidebar.selectbox("Education 🎓",  df['Education'].unique())
    codes_java = st.sidebar.checkbox("Codes in Java ☕")
    codes_python = st.sidebar.checkbox("Codes in Python 🐍")
    codes_sql = st.sidebar.checkbox("Codes in SQL 🗄️")
    codes_go = st.sidebar.checkbox("Codes in Go 🐹")

    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        'Age': [age],
        'Years_Coding': [age],  # Assuming Years_Coding is the same as Age for simplicity
        'Education': [education],
        'Codes_In_JAVA': [int(codes_java)],
        'Codes_In_Python': [int(codes_python)],
        'Codes_In_SQL': [int(codes_sql)],
        'Codes_In_GO': [int(codes_go)],
        'Country': [country],
    })

    # One-hot encode the  country
    input_data = pd.get_dummies(input_data, columns=['Country'], drop_first=True)
    # Align the columns of the input data with the columns used during training.
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)


    # Make prediction
    if st.button("Predict Salary 💰"): # Added emoji to the button
        prediction = model.predict(input_data)
        st.success(f"Predicted Salary: 💵 ${prediction[0]:.2f} 💵")  # Added emoji to the output



if __name__ == '__main__':
    main()

