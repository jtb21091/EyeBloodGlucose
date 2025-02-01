import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
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

def train_model():
    if not os.path.exists(labels_file):
        print("Error: Data file not found.")
        return
    
    df = pd.read_csv(labels_file)
    
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
    
    df = remove_outliers(df)
    
    if len(df) < 5:
        print(f"Not enough data to train (found {len(df)} rows). Need at least 5 rows.")
        return
    
    df["blood_glucose"] = df["blood_glucose"].astype(float)
    X = df.drop(columns=["blood_glucose"])
    y = df["blood_glucose"]
    
    # Ensure all features are numeric
    non_numeric_cols = X.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        print(f"⚠️ Warning: Found non-numeric columns: {list(non_numeric_cols)}")
        X = X.drop(columns=non_numeric_cols)  # Drop non-numeric columns
    
    # Fill any remaining NaN values with 0
    X.fillna(0, inplace=True)
    
    test_size = 0.2 if len(df) > 10 else 0.0
    if test_size == 0.0:
        X_train, y_train = X, y
        X_val, y_val = X, y
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Neural Network": MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42, verbose=True, early_stopping=True, validation_fraction=0.2, n_iter_no_change=10),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Support Vector Regression": SVR(kernel='rbf', C=100, gamma=0.1)
    }
    
    best_model = None
    best_score = float('-inf')
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        mse = mean_squared_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)
        print(f"{name} - MSE: {mse:.10f}, R² Score: {r2:.5f}")
        
        if name == "Neural Network":
            plt.plot(model.loss_curve_, label="Training Loss")
            if hasattr(model, "validation_scores_"):
                plt.plot(model.validation_scores_, label="Validation Loss")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.title("Neural Network Training and Validation Loss Curve")
            plt.legend()
            plt.show()
        
        if r2 > best_score:
            best_model = model
            best_score = r2
    
    print(f"Best Model: {best_model.__class__.__name__} with R² Score: {best_score:.5f}")
    
    joblib.dump(best_model, model_file)
    print(f"Best model saved to {model_file}")

if __name__ == "__main__":
    train_model()
