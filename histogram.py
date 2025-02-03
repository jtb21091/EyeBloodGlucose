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
    
    # Determine bin edges (rounded to the nearest 10)
    min_bin = int(np.floor(min(blood_glucose) / 10) * 10)
    max_bin = int(np.ceil(max(blood_glucose) / 10) * 10)
    bins = np.arange(min_bin, max_bin + 10, 10)  # Bins in increments of 10

    # Plot histogram
    plt.figure(figsize=(10, 6))
    count, bins, _ = plt.hist(blood_glucose, bins=bins, edgecolor='black', alpha=0.7, label="Histogram")

    # Generate x values for the normal distribution curve
    x = np.linspace(min(blood_glucose), max(blood_glucose), 100)
    
    # Compute normal distribution (bell curve) and scale it to match histogram frequency
    bin_width = bins[1] - bins[0]  # Width of each bin
    pdf = norm.pdf(x, mean_value, std_dev) * sum(count) * bin_width  # Scale to match frequency

    # Plot the bell curve
    plt.plot(x, pdf, color='green', linewidth=2, label="Bell Curve (Normal Dist.)")
    
    # Add vertical lines for mean and median
    plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.0f}')
    
    median_value = blood_glucose.median()
    plt.axvline(median_value, color='blue', linestyle='-.', linewidth=2, label=f'Median: {median_value:.0f}')
    
    # Set x-axis ticks to show bin edges in whole numbers
    plt.xticks(bins, [f"{int(b)}" for b in bins])  # Convert to integers
    
    plt.title("Histogram of Blood Glucose with Bell Curve")
    plt.xlabel("Blood Glucose (mg/dL)")
    plt.ylabel("Frequency")  # Changed from Density to Frequency
    plt.grid(True)
    plt.legend()
    plt.show()

else:
    print("Column 'blood_glucose' not found in the CSV file.")
