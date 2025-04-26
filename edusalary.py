import pickle
import numpy as np

# Load the trained model
with open('kaggle_edu_model_2022.pkl', 'rb') as f:
    model = pickle.load(f)

def main():
    print("\n=== Salary Prediction App ===\n")

    # Collect user input (numerical codes expected)
    try:
        duration = int(input("Enter Duration (time spent on survey, seconds): "))
        age = int(input("Enter Age (encoded as integer): "))
        gender = int(input("Enter Gender (encoded as integer): "))
        country = int(input("Enter Country (encoded as integer): "))
        education = int(input("Enter Education level (encoded as integer): "))
        
        # Add any more fields if your model expects them
        # Right now I'm assuming only a few important ones

        # Build the input vector
        input_data = np.array([[duration, age, gender, country, education]])

        # Predict salary
        predicted_salary = model.predict(input_data)

        print(f"\nPredicted Salary: ${predicted_salary[0]:,.2f}\n")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
