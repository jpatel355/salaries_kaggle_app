import streamlit as st
import pandas as pd
import pickle
import os  # Import the os module
from sklearn.model_selection import train_test_split # Import train_test_split

def main():
    st.title("💰 Salary Prediction App 💰")  # Added a title with emojis

    # Load the pickled model
    try:
        with open('salary_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError:
        st.error(f"Error: Model file 'salary_model.pkl' not found. Please make sure the file is in the same directory as the script, or provide the correct path.")
        return  

    # Load the training data to get the country list and column order
    # Use os.path.join to handle file path correctly
    csv_file_path = os.path.join('.', 'kaggle_survey_2022_responses.csv')  # Assuming the CSV is in the same directory
    if not os.path.exists(csv_file_path):
        st.error(f"Error: CSV file not found at {csv_file_path}.  Please make sure the file is in the same directory as the script, or provide the correct path.")
        return  # Stop if the file is not found

    try:
        df = pd.read_csv(csv_file_path)
        df = df.drop(index=0).reset_index(drop=True)
         # Step 4: Rename key columns
        df.rename(columns={
            'Q2': 'Age',
            'Q3': 'Gender',
            'Q4': 'Country',
            'Q5': 'Student Status',
            'Q8': 'Education'
        }, inplace=True)
        categorical_cols = ['Country']
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['Years_Coding'] = df['Age']
        X = df[categorical_cols]
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)  # Moved inside the try block
    except Exception as e:
        st.error(f"Error reading or processing data: {e}. Please check the data file and format.")
        return

    # Create input widgets with a more organized layout
    st.sidebar.header("Input Features ⚙️")  # Added a sidebar header

    # Use the pre-processed data to get country list.
    age = st.sidebar.slider("Age 🎂", 18, 65, 30)
    country = st.sidebar.selectbox("Country 🌍", X_train['Country'].unique()) # Changed to X_train
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
        try:
            prediction = model.predict(input_data)
            st.success(f"Predicted Salary: 💵 ${prediction[0]:.2f} 💵")  # Added emoji to the output
        except Exception as e:
            st.error(f"Error during prediction: {e}.  Please check the input data and model.")
            return

if __name__ == '__main__':
    main()
