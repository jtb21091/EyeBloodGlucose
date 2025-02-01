import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file (adjust the path if needed)
df = pd.read_csv("eye_glucose_data/labels.csv")

# Check that the column exists
if "blood_glucose" in df.columns:
    # Extract the blood_glucose column
    blood_glucose = df["blood_glucose"]
    
    # Calculate the mean
    mean_value = blood_glucose.mean()
    print("Mean Blood Glucose Value:", mean_value)
    
    # Plot a histogram
    plt.figure(figsize=(10, 6))
    plt.hist(blood_glucose, bins=30, edgecolor='black', alpha=0.7)
    
    # Add a vertical line at the mean
    plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_value:.2f}')
    
    plt.title("Histogram of Blood Glucose")
    plt.xlabel("Blood Glucose (mg/dL)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.legend()
    plt.show()

else:
    print("Column 'blood_glucose' not found in the CSV file.")
