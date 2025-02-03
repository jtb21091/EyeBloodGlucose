import pandas as pd
import numpy as np

# Load the CSV file
file_path = "eye_glucose_data/labels.csv"
df = pd.read_csv(file_path)

# Identify numeric columns excluding "blood_glucose" and "filename"
excluded_columns = {"blood_glucose", "filename"}
numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in excluded_columns]

# Compute Z-scores and filter values with Z-score > 6
z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
outliers = (z_scores > 6)

# Extract rows where any column has a Z-score > 6
outlier_values = df[outliers.any(axis=1)]

# Save the outlier values to a CSV file named "sixsigma.csv"
output_file_path = "sixsigma.csv"
df[outliers].dropna(how='all').to_csv(output_file_path, index=False)
