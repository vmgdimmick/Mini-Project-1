import pandas as pd
import pickle

from tensorflow.keras.models import load_model

def predict_job_satisfaction(new_employee_data):
    """
    1) Loads the saved model (my_model.h5), scaler (scaler.pkl), and columns (columns.txt).
    2) Converts the new data dict to a DataFrame.
    3) One-hot-encodes it exactly like training, ensuring we have all the same columns.
    4) Scales numeric columns with the loaded scaler.
    5) Calls model.predict(...) and returns the predicted JobSatisfaction.
    """

    # Load the final columns from training
    with open("columns.txt", "r") as f:
        final_columns = f.read().splitlines()  # list of column names

    # Load the scaler
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    #Load the trained Keras model
    model = load_model("neuralnet1.keras")

    # Convert the new data (dict) to a DataFrame
    X_new = pd.DataFrame([new_employee_data])
    # E.g. X_new columns might be: [BusinessTravel, DistanceFromHome, YearsAtCompany, Age]

    # Identify which columns were categorical in training
    categorical_cols = ["BusinessTravel"]  # same as training
    numeric_cols = ["DistanceFromHome", "YearsAtCompany", "Age"]  # same as training

    # One-hot-encode the new data's categorical columns
    X_new = pd.get_dummies(X_new, columns=categorical_cols, drop_first=True)

    #  Add any missing columns that were in final_columns but not in X_new
    for col in final_columns:
        if col not in X_new.columns:
            X_new[col] = 0

    # Ensure the exact same column order
    X_new = X_new[final_columns]

    # Scale numeric columns with the loaded scaler
    X_new[numeric_cols] = scaler.transform(X_new[numeric_cols])

    # Predict with the model
    prediction = model.predict(X_new)

    # Return the single numeric value (assuming one row)
    return prediction[0][0]


if __name__ == "__main__":
    # Example usage:
    # python predict.py
# creating test cases
    test_cases = [
        {"BusinessTravel": "Travel_Frequently", "DistanceFromHome": 1, "YearsAtCompany": 1, "Age": 22},
        {"BusinessTravel": "Travel_Rarely", "DistanceFromHome": 50, "YearsAtCompany": 30, "Age": 60},
        {"BusinessTravel": "Travel_Frequently", "DistanceFromHome": 20, "YearsAtCompany": 15, "Age": 35},
        {"BusinessTravel": "Travel_Rarely", "DistanceFromHome": 5, "YearsAtCompany": 5, "Age": 25},
        {"BusinessTravel": "Travel_Frequently", "DistanceFromHome": 100, "YearsAtCompany": 40, "Age": 65},
        {"BusinessTravel": "Travel_Rarely", "DistanceFromHome": 30, "YearsAtCompany": 10, "Age": 45},
        {"BusinessTravel": "Travel_Frequently", "DistanceFromHome": 75, "YearsAtCompany": 25, "Age": 55},
        {"BusinessTravel": "Travel_Rarely", "DistanceFromHome": 10, "YearsAtCompany": 2, "Age": 23},
        {"BusinessTravel": "Travel_Frequently", "DistanceFromHome": 60, "YearsAtCompany": 35, "Age": 58},
        {"BusinessTravel": "Travel_Rarely", "DistanceFromHome": 3, "YearsAtCompany": 8, "Age": 29}
    ]
# Find prediction for every test case
    for i, new_employee_dict in enumerate(test_cases, start=1):
        predicted_satisfaction = predict_job_satisfaction(new_employee_dict)
        print(f"Test Case {i}: {new_employee_dict}")
        print(f"Predicted Job Satisfaction: {predicted_satisfaction}\n")
