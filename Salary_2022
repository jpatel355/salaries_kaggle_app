
import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open('Salary2022_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the user input function
def user_input_features():
    age = st.selectbox('Age', ['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-69', '70+'])
    gender = st.selectbox('Gender', ['Man', 'Woman', 'Nonbinary', 'Prefer not to say'])
    country = st.text_input('Country')
    q7_1 = st.checkbox('Q7_1')
    q7_2 = st.checkbox('Q7_2')
    q7_3 = st.checkbox('Q7_3')
    q7_4 = st.checkbox('Q7_4')
    q7_5 = st.checkbox('Q7_5')
    q7_6 = st.checkbox('Q7_6')
    q7_7 = st.checkbox('Q7_7')
    
    data = {
        'Age': age,
        'Gender': gender,
        'Country': country,
        'Q7_1': q7_1,
        'Q7_2': q7_2,
        'Q7_3': q7_3,
        'Q7_4': q7_4,
        'Q7_5': q7_5,
        'Q7_6': q7_6,
        'Q7_7': q7_7
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Main function to run the app
def main():
    st.title("Student Status Prediction")
    st.write("Predict whether a user is a student based on their input data.")
    
    input_df = user_input_features()
    
    # Preprocess the input data
    input_df = pd.get_dummies(input_df)
    
    # Ensure all columns are present
    missing_cols = set(X.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[X.columns]
    
    # Make predictions
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    st.subheader('Prediction')
    st.write('Student' if prediction[0] == 1 else 'Not a Student')
    
    st.subheader('Prediction Probability')
    st.write(prediction_proba)

if __name__ == '__main__':
    main()
