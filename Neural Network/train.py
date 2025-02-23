import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf

def train_model(data_file):
    """
    1) Loads CSV data
    2) One-hot-encodes categorical columns
    3) Splits into train/test
    4) Scales numeric columns
    5) Trains a neural network
    6) Saves model, scaler, and final column list
    """
    
    # Step A: Load the dataset
   
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

    
    # Step B: One-hot-encode the categorical columns
    
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Save the final columns after encoding (for alignment in inference)
    final_columns = X.columns.tolist()

    
    # Step C: Train/test split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    
    # Step D: Scale numeric columns
    
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    
    # Build and train the model

    # A simple feedforward neural network for regression with one hidden layer
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(1, activation='linear'))  # single numeric output
    model.compile(optimizer='adam', loss='mse')
    
    # Custom early stopping callback to halt training if validation loss stops improving
    class EarlyStoppingMonitor(tf.keras.callbacks.Callback):
        def __init__(self, patience=3):
            super().__init__()
            self.patience = patience
            self.best_weights = None
            self.best_epoch = 0
            self.best_loss = np.inf
            self.wait = 0
            self.stopped_epoch = 0
            self.losses = []
            self.val_losses = []
            self.monitoring = []
        #  Monitors validation loss at the end of each epoch and stops training if no improvement is seen for a set patience period
        def on_epoch_end(self, epoch, logs=None):
            current_loss = logs.get('val_loss')
            self.losses.append(logs.get('loss'))
            self.val_losses.append(current_loss)

            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.wait = 0
                self.best_epoch = epoch
                self.best_weights = self.model.get_weights()
                self.monitoring.append('✓ New best model!')
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.monitoring.append(f'⛔ Stopping! No improvement for {self.patience} epochs')
                    self.model.stop_training = True
                else:
                    self.monitoring.append(f'⚠️ No improvement: patience {self.wait}/{self.patience}')
                    
        # Plots training and validation loss over epochs, marking the best model and early stopping point
        def plot_training(self):
            plt.figure(figsize=(12, 6))
            epochs = range(1, len(self.losses) + 1)

            # Plot losses
            plt.plot(epochs, self.losses, 'b-', label='Training Loss')
            plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')

            # Mark the best epoch
            plt.axvline(x=self.best_epoch + 1, color='g', linestyle='--',
                        label=f'Best Model (Epoch {self.best_epoch + 1})')

            # Highlight stopping point
            if self.stopped_epoch:
                plt.axvline(x=self.stopped_epoch + 1, color='r', linestyle='--',
                            label=f'Early Stopping (Epoch {self.stopped_epoch + 1})')

            plt.title('Training Progress with Early Stopping')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            # Print monitoring log
            print("\nTraining Monitor Log:")
            for epoch, message in enumerate(self.monitoring, 1):
                print(f"Epoch {epoch}: {message}")

    # Defining process for early stopping (monitoring and stopping)
    monitor = EarlyStoppingMonitor(patience=3)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the neural network model with training data and monitor its performance
    history = model.fit(
        X_train, y_train,  # Training data (features and target)
        validation_data=(X_test, y_test),  # Validation data to monitor generalization performance
        callbacks=[monitor, early_stopping],  # Custom callbacks for tracking progress and stopping early if needed
        epochs=50,  # Maximum number of training iterations
        batch_size=32,  # Number of samples processed before updating model weights
        verbose=1  # Display training progress
    )
    
    # Plot the training and validation loss over epochs to visualize model performance
    monitor.plot_training()  # Generate the loss vs. epoch plot with early stopping markers
    plt.show()  # Display the plot

    # Evaluate
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Training MSE: {train_loss:.4f}")
    print(f"Final Testing MSE:  {test_loss:.4f}")
    
    # Save model, scaler, and column info
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
