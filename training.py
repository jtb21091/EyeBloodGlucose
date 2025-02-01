import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

labels_file = "eye_glucose_data/labels.csv"
model_file = "eye_glucose_model.pkl"

def train_model():
    if not os.path.exists(labels_file):
        print("Error: Data file not found.")
        return
    
    # Load the dataset
    df = pd.read_csv(labels_file)
    df.dropna(inplace=True)  # Remove rows with missing values
    
    # Extract features and target
    X = df[["pupil_size", "sclera_redness"]]
    y = df["blood_glucose"].astype(float)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model trained. MSE: {mse:.2f}")
    
    # Save the trained model
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    train_model()
