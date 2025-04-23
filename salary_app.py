import streamlit as st
import pickle
import os

def main():
    st.title("💰 Salary Prediction App 💰")

    # Correct file path to the model
    model_file_path = os.path.join('.', 'kaggle2022_model (3).pkl')  # Use the correct model file name

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

    # Load the dataset
    csv_file_path = os.path.join('.', 'kaggle_survey_2022_responses.csv')  # Make sure this CSV is in the correct directory
    if not os.path.exists(csv_file_path):
        st.error(f"Error: CSV file not found at {csv_file_path}.")
        return  

    try:
        df = pd.read_csv(csv_file_path)
        df = df.drop(index=0).reset_index(drop=True)
        df.rename(columns={
            'Q2': 'Age',
            'Q3': 'Gender',
            'Q4': 'Country',
            'Q5': 'Student Status',
            'Q8': 'Education'
        }, inplace=True)
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['Years_Coding'] = df['Age']
        categorical_cols = ['Country']
        X = df[categorical_cols]
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    except Exception as e:
        st.error(f"Error processing dataset: {e}")
        return

    st.sidebar.header("Input Features ⚙️")
    age = st.sidebar.slider("Age 🎂", 18, 65, 30)

    # Extract countries from model columns
    available_countries = [col.replace("Country_", "") for col in model_columns if col.startswith("Country_")]
    country = st.sidebar.selectbox("Country 🌍", sorted(available_countries))
    
    education = st.sidebar.selectbox("Education 🎓", df['Education'].dropna().unique())
    codes_java = st.sidebar.checkbox("Codes in Java ☕")
    codes_python = st.sidebar.checkbox("Codes in Python 🐍")
    codes_sql = st.sidebar.checkbox("Codes in SQL 🗄️")
    codes_go = st.sidebar.checkbox("Codes in Go 🐹")

    input_data = pd.DataFrame({
        'Age': [age],
        'Years_Coding': [age],
        'Education': [education],
        'Codes_In_JAVA': [int(codes_java)],
        'Codes_In_Python': [int(codes_python)],
        'Codes_In_SQL': [int(codes_sql)],
        'Codes_In_GO': [int(codes_go)],
        'Country': [country],
    })

    # One-hot encode and reindex input data
    input_data = pd.get_dummies(input_data, columns=['Country'], prefix='Country')
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Predict salary when the button is clicked
    if st.button("Predict Salary 💰"):
        try:
            prediction = model.predict(input_data)
            st.success(f"Predicted Salary: 💵 ${prediction[0]:.2f} 💵")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
