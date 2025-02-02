import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Load the CSV file (adjust the path if needed)
df = pd.read_csv("eye_glucose_data/labels.csv")

# Check that the column exists
if "blood_glucose" in df.columns:
    # Extract the blood_glucose column
    blood_glucose = df["blood_glucose"]
    
    # Calculate the mean and standard deviation
    mean_value = blood_glucose.mean()
    std_dev = blood_glucose.std()
    
    print("Mean Blood Glucose Value:", mean_value)
    print("Standard Deviation:", std_dev)
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    count, bins, _ = plt.hist(blood_glucose, bins=30, density=True, edgecolor='black', alpha=0.7, label="Histogram")

    # Generate x values for the normal distribution curve
    x = np.linspace(min(blood_glucose), max(blood_glucose), 100)
    
    # Compute normal distribution (bell curve)
    pdf = norm.pdf(x, mean_value, std_dev)
    
    # Plot the bell curve
    plt.plot(x, pdf, color='green', linewidth=2, label="Bell Curve (Normal Dist.)")
    
    # Add vertical lines for mean and median
    plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
    
    median_value = blood_glucose.median()
    plt.axvline(median_value, color='blue', linestyle='-.', linewidth=2, label=f'Median: {median_value:.2f}')
    
    plt.title("Histogram of Blood Glucose with Bell Curve")
    plt.xlabel("Blood Glucose (mg/dL)")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    plt.show()

else:
    print("Column 'blood_glucose' not found in the CSV file.")
