import pandas as pd
import statsmodels.api as sm

# Load the dataset
file_path = "eye_glucose_data/labels.csv"
df = pd.read_csv(file_path)

# Drop the 'filename' column as requested
df = df.drop(columns=['filename'])

# Drop columns with too many NaN values and fill remaining NaNs with mean values
df = df.dropna(axis=1, thresh=int(0.7 * len(df)))  # Drop columns with more than 30% missing values
df = df.fillna(df.mean())  # Fill remaining NaNs with column means

# Separate independent and dependent variables
X = df.drop(columns=['blood_glucose'])
y = df['blood_glucose']

# Add a constant for the regression model
X = sm.add_constant(X)

# Fit an Ordinary Least Squares (OLS) model
model = sm.OLS(y, X).fit()

# Extract p-values and other key statistics
p_values = model.pvalues
coefficients = model.params
std_errors = model.bse

# Create a DataFrame for variable metrics
metrics_df = pd.DataFrame({
    'Coefficient': coefficients,
    'P-Value': p_values,
    'Standard Error': std_errors,
})

# Reset index to include variable names
metrics_df = metrics_df.reset_index().rename(columns={'index': 'Variable'})

# Exclude the constant from ranking
non_const_df = metrics_df[metrics_df['Variable'] != 'const'].copy()

# Rank variables by absolute coefficient values
non_const_df['Rank'] = non_const_df['Coefficient'].abs().rank(ascending=False)

# Sort the dataframe by rank
non_const_df = non_const_df.sort_values(by="Rank")

# Re-add the constant to the dataframe but exclude it from ranking
const_row = pd.DataFrame({
    'Variable': ['const'],
    'Coefficient': model.params['const'],
    'P-Value': model.pvalues['const'],
    'Standard Error': model.bse['const'],
    'Rank': None  # No rank for constant
}, index=[-1])

# Append the constant row back to the metrics dataframe
metrics_df = pd.concat([const_row, non_const_df]).reset_index(drop=True)

# Display the results
# Save results to a CSV file
metrics_df.to_csv("blood_glucose_prediction_metrics.csv", index=False)

# Print the DataFrame to check output
print(metrics_df)

