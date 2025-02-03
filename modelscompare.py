import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('modelscompare.csv')

# Remove any leading/trailing whitespace from column names
df.columns = df.columns.str.strip()

# Debug: Print the column names to ensure they are correct
print("CSV columns:", df.columns.tolist())

# Define the list of metrics to plot
metrics = ['R2', 'MSE', 'MAE', 'MARD', 'Sigma Level']

# Get the list of model names
models = df['models'].tolist()

# Create a list of colors:
# "green" for CGM Benchmark and "red" for any other model.
bar_colors = ['green' if model == 'CGM Benchmark' else 'red' for model in models]

# Create subplots: one subplot per metric
fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))

# Loop through each metric and create a bar chart
for ax, metric in zip(axes, metrics):
    # Get the values for the current metric
    values = df[metric].tolist()
    
    # Plot the bar chart for this metric using the specified colors
    bars = ax.bar(models, values, color=bar_colors)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    
    # Annotate each bar with its value
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # vertical offset in points
                    textcoords='offset points',
                    ha='center', va='bottom')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
