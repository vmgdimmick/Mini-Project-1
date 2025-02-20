import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

def train_model(data_file):
    """
    1) Loads CSV data
    2) One-hot-encodes categorical columns
    3) Splits into train/test
    4) Scales numeric columns
    5) Trains a neural network
    6) Saves model, scaler, and final column list
    """
    # ---------------------------------------
    # Step A: Load the dataset
    # ---------------------------------------
    df = pd.read_csv(data_file)
    # E.g. columns: [BusinessTravel, DistanceFromHome, YearsAtCompany, Age, JobSatisfaction]
    # Adjust names as needed
    print(df.info())  # Check dtypes and non-null counts
    print(df.describe(include='all'))  # Check for weird min/max
    print(df.isna().sum())  # Check how many NaNs per column

    df.dropna(inplace=True)
    # Separate target
    y = df["JobSatisfaction"]
    X = df.drop("JobSatisfaction", axis=1)

    # Identify numeric vs. categorical
    # (Adjust as needed for your data)
    numeric_cols = ["DistanceFromHome", "YearsAtCompany", "Age"]
    categorical_cols = ["BusinessTravel"]

    # ---------------------------------------
    # Step B: One-hot-encode the categorical columns
    # ---------------------------------------
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Save the final columns after encoding (for alignment in inference)
    final_columns = X.columns.tolist()

    # ---------------------------------------
    # Step C: Train/test split
    # ---------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------------------
    # Step D: Scale numeric columns
    # ---------------------------------------
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # ---------------------------------------
    # Step E: Build and train the model
    # ---------------------------------------
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(1, activation='linear'))  # single numeric output
    model.compile(optimizer='adam', loss='mse')

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # Evaluate
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Training MSE: {train_loss:.4f}")
    print(f"Final Testing MSE:  {test_loss:.4f}")

    # ---------------------------------------
    # Step F: Save model, scaler, and column info
    # ---------------------------------------
    model.save("neuralnet1.keras")
    print("Model saved to neuralnet1.keras")

    # Save the scaler
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Scaler saved to scaler.pkl")

    # Save final columns list
    with open("columns.txt", "w") as f:
        f.write("\n".join(final_columns))
    print("Column list saved to columns.txt")


if __name__ == "__main__":
    # Example usage:
    # python train.py
    data_csv_path = "filtered_data.csv"  # Replace with your actual CSV
    train_model(data_csv_path)