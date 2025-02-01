import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

labels_file = "eye_glucose_data/labels.csv"
model_file = "eye_glucose_model.pkl"

def train_model():
    if not os.path.exists(labels_file):
        print("Error: Data file not found.")
        return
    
    df = pd.read_csv(labels_file)
    
    # Fill missing values instead of dropping them
    df.fillna({"vein_prominence": 0.0, "pupil_response_time": 0.2, "predicted_glucose": 0.0}, inplace=True)
    
    print(f"Dataset size after cleaning: {len(df)} rows")
    
    if len(df) < 5:
        print(f"Not enough data to train (found {len(df)} rows). Need at least 5 rows.")
        return
    
    df["blood_glucose"] = df["blood_glucose"].astype(float)
    X = df[["pupil_size", "sclera_redness", "vein_prominence", "pupil_response_time"]]
    y = df["blood_glucose"]
    
    test_size = 0.2 if len(df) > 10 else 0.0  # Only split if enough data
    if test_size == 0.0:
        X_train, y_train = X, y
        X_test, y_test = X, y  # Evaluate on the same data
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)  # Calculate R² score

    print(f"Model trained. MSE: {mse:.10f}")
    print(f"Model trained. R² Score: {r2:.5f}")  # Display R² Score
    
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    train_model()
