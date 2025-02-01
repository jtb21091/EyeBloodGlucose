import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file (adjust the path if needed)
df = pd.read_csv("eye_glucose_data/labels.csv")

# Check that the column exists
if "blood_glucose" in df.columns:
    # Extract the blood glucose column
    blood_glucose = df["blood_glucose"]

    # Plot a histogram
    plt.figure(figsize=(10, 6))
    plt.hist(blood_glucose, bins=30, edgecolor='black')
    plt.title("Histogram of Blood Glucose")
    plt.xlabel("Blood Glucose (mg/dL)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
else:
    print("Column 'blood_glucose' not found in the CSV file.")
