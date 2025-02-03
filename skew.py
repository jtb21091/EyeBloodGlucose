import pandas as pd
import numpy as np

# Load the CSV file (adjust the path if needed)
df = pd.read_csv("eye_glucose_data/labels.csv")

# Check if 'blood_glucose' column exists
if "blood_glucose" in df.columns:
    # Print statistical description
    print(df["blood_glucose"].describe())
    
    # Print skewness and kurtosis
    print("Skewness:", df["blood_glucose"].skew())  # > 0 means right-skewed
    print("Kurtosis:", df["blood_glucose"].kurtosis())  # Measures tail weight
    
    # Log transformation of blood glucose values
    df["log_blood_glucose"] = np.log(df["blood_glucose"])
    print("Log-transformed Blood Glucose:")
    print(df["log_blood_glucose"].describe())
else:
    print("Column 'blood_glucose' not found in the CSV file.")
