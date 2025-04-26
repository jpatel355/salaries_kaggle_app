import pickle
import numpy as np

# Load the trained model
with open('kaggle_edu_model_2022.pkl', 'rb') as f:
    model = pickle.load(f)

def get_input(prompt, default):
    """Helper function: gets input or uses default if blank."""
    user_input = input(f"{prompt} [{default}]: ")
    if user_input.strip() == "":
        return default
    try:
        return int(user_input)
    except ValueError:
        print("Invalid input. Using default value.")
        return default

def main():
    print("\n" + "="*40)
    print("        Salary Prediction App")
    print("="*40 + "\n")

    try:
        # Collect user inputs
        duration = get_input("Enter Duration (time spent on survey, seconds)", 300)
        age = get_input("Enter Age (encoded as integer)", 4)
        gender = get_input("Enter Gender (encoded as integer)", 1)
        country = get_input("Enter Country (encoded as integer)", 10)
        education = get_input("Enter Education level (encoded as integer)", 2)

        # Build the input vector
        input_data = np.array([[duration, age, gender, country, education]])

        # Predict salary
        predicted_salary = model.predict(input_data)

        # Output result
        print("\n" + "-"*40)
        print(f"🎯 Predicted Salary: ${predicted_salary[0]:,.2f}")
        print("-"*40 + "\n")

    except Exception as e:
        print("\n⚠️ Error occurred during prediction.")
        print(f"Details: {e}\n")

if __name__ == "__main__":
    main()
