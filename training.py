import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

labels_file = "eye_glucose_data/labels.csv"
model_file = "eye_glucose_model.pkl"

def remove_outliers(df):
    """Removes outliers using the Interquartile Range (IQR)."""
    Q1 = df["blood_glucose"].quantile(0.25)
    Q3 = df["blood_glucose"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df["blood_glucose"] >= lower_bound) & (df["blood_glucose"] <= upper_bound)].copy()
    print(f"Outliers removed. Dataset reduced from {len(df)} to {len(df_filtered)} rows.")
    return df_filtered

def train_model(use_neural_network=False):
    if not os.path.exists(labels_file):
        print("Error: Data file not found.")
        return
    
    df = pd.read_csv(labels_file)
    
    # Fill missing values
    df.fillna({
        "vein_prominence": 0.0, 
        "pupil_response_time": 0.2, 
        "ir_intensity": 0.0, 
        "pupil_circularity": 1.0, 
        "scleral_vein_density": 0.0, 
        "blink_rate": 0.0, 
        "ir_temperature": 0.0, 
        "tear_film_reflectivity": 0.0, 
        "pupil_dilation_rate": 0.5, 
        "sclera_color_balance": 1.0, 
        "vein_pulsation_intensity": 0.0
    }, inplace=True)
    
    # Remove outliers
    df = remove_outliers(df)

    print(f"Dataset size after cleaning: {len(df)} rows")
    
    if len(df) < 5:
        print(f"Not enough data to train (found {len(df)} rows). Need at least 5 rows.")
        return
    
    df["blood_glucose"] = df["blood_glucose"].astype(float)
    X = df[[ "sclera_redness", "vein_prominence", "pupil_response_time", "ir_intensity", "pupil_circularity", "scleral_vein_density", "blink_rate", "ir_temperature", "tear_film_reflectivity", "pupil_dilation_rate", "sclera_color_balance", "vein_pulsation_intensity"]]
    y = df["blood_glucose"]
    
    # Remove constant features
    constant_features = [col for col in X.columns if X[col].nunique() == 1]
    if constant_features:
        print(f"Removing constant features: {constant_features}")
        X = X.drop(columns=constant_features).copy()  # Use .copy() to avoid SettingWithCopyWarning
    
    # Check for NaN values before training
    if X.isnull().sum().sum() > 0:
        print("⚠️ Warning: Training data contains NaN values! Filling missing values now.")
        X.fillna(0, inplace=True)

    test_size = 0.2 if len(df) > 10 else 0.0  # Only split if enough data
    if test_size == 0.0:
        X_train, y_train = X, y
        X_test, y_test = X, y  # Evaluate on the same data
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    if use_neural_network:
        model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42, verbose=True)
        model.fit(X_train, y_train)
        
        # Collect loss history
        training_loss = model.loss_curve_
        plt.plot(training_loss)
        plt.xlabel("Iterations")
        plt.ylabel("Training Loss")
        plt.title("Neural Network Training Loss Curve")
        plt.show()
        
    else:
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
    train_model(use_neural_network=True)  # Set to True to use MLP Neural Network
