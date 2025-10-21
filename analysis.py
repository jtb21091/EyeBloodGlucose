import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Read the data
df = pd.read_csv('eye_glucose_data/labels.csv')

# Debug: Print available columns
print("Available columns in CSV:")
print(df.columns.tolist())
print()

# Check if blood_glucose column exists
if 'blood_glucose' not in df.columns:
    print("❌ ERROR: 'blood_glucose' column not found in CSV!")
    print("\nThis might be because:")
    print("  1. You're using an old CSV with different column names")
    print("  2. The CSV header is corrupted")
    print("\nPlease check your eye_glucose_data/labels.csv file.")
    print("Expected columns should include: filename, blood_glucose, pupil_size, etc.")
    exit(1)

# Filter out rows without glucose values
df_valid = df[df['blood_glucose'].notna() & (df['blood_glucose'] != '')].copy()
df_valid['blood_glucose'] = pd.to_numeric(df_valid['blood_glucose'], errors='coerce')
df_valid = df_valid.dropna(subset=['blood_glucose'])

print(f"Total samples: {len(df)}")
print(f"Valid samples with glucose: {len(df_valid)}")
print(f"Missing glucose: {len(df) - len(df_valid)}")

if len(df_valid) == 0:
    print("\n⚠️  No valid glucose data found. Cannot perform analysis.")
    print("Please add blood glucose values to your CSV file.")
    exit(0)

print(f"\nGlucose range: {df_valid['blood_glucose'].min():.0f} - {df_valid['blood_glucose'].max():.0f} mg/dL")
print(f"Mean glucose: {df_valid['blood_glucose'].mean():.1f} mg/dL")
print(f"Std dev: {df_valid['blood_glucose'].std():.1f} mg/dL")

# Updated feature columns (with new features and renamed columns)
feature_columns = [
    'pupil_size', 'sclera_redness', 'vein_prominence', 
    'capture_duration', 'ir_intensity', 'scleral_vein_density',
    'ir_temperature', 'tear_film_reflectivity',
    'sclera_color_balance', 'vein_pulsation_intensity', 'birefringence_index',
    'lens_clarity_score', 'sclera_yellowness', 'vessel_tortuosity',
    'image_quality_score'
]

# Also check for old column names for backward compatibility
old_column_mapping = {
    'pupil_response_time': 'capture_duration',
    'pupil_dilation_rate': 'pupil_size'  # Was duplicate
}

# Use available columns only
available_features = [col for col in feature_columns if col in df_valid.columns]

# Add old columns if they exist and new ones don't
for old_col, new_col in old_column_mapping.items():
    if old_col in df_valid.columns and new_col not in available_features:
        available_features.append(old_col)
        print(f"Note: Using old column name '{old_col}' (should be '{new_col}' in new version)")

if len(available_features) == 0:
    print("\n⚠️  No feature columns found for analysis.")
    exit(0)

if len(df_valid) < 3:
    print("\n⚠️  Need at least 3 samples to perform correlation analysis.")
    print(f"You currently have {len(df_valid)} samples. Keep collecting data!")
    exit(0)

correlations = {}
p_values = {}

print("\n" + "="*70)
print("CORRELATION ANALYSIS WITH BLOOD GLUCOSE")
print("="*70)

for feature in available_features:
    if feature in df_valid.columns:
        # Remove NaN values for this feature
        valid_pairs = df_valid[['blood_glucose', feature]].dropna()
        
        if len(valid_pairs) > 2:
            corr, p_val = stats.pearsonr(valid_pairs['blood_glucose'], valid_pairs[feature])
            correlations[feature] = corr
            p_values[feature] = p_val
            
            # Determine significance
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            
            print(f"\n{feature:.<35} r={corr:>7.4f}  p={p_val:.4f} {sig}")
            print(f"  {'└─ Sample size: ' + str(len(valid_pairs))}")

if len(correlations) == 0:
    print("\n⚠️  Not enough data to calculate correlations.")
    exit(0)

# Sort by absolute correlation
sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print("\n" + "="*70)
print("RANKED BY CORRELATION STRENGTH (absolute value)")
print("="*70)

for i, (feature, corr) in enumerate(sorted_corr, 1):
    p_val = p_values[feature]
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    direction = "↑" if corr > 0 else "↓"
    print(f"{i:2d}. {feature:.<35} {direction} {abs(corr):.4f} ({sig})")

print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns=not significant")

# Highlight new features
new_features = ['lens_clarity_score', 'sclera_yellowness', 'vessel_tortuosity', 'image_quality_score']
new_features_present = [f for f in new_features if f in correlations]

if new_features_present:
    print("\n" + "="*70)
    print("NEW FEATURES ADDED")
    print("="*70)
    for feature in new_features_present:
        corr = correlations[feature]
        p_val = p_values[feature]
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"{feature:.<35} r={corr:>7.4f} ({sig})")

# Create visualization
n_features = len(available_features)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
fig.suptitle('Feature Correlations with Blood Glucose', fontsize=16, fontweight='bold')

# Flatten axes for easier iteration
if n_rows == 1:
    axes = axes.reshape(1, -1)
axes_flat = axes.flatten()

for idx, feature in enumerate(available_features):
    ax = axes_flat[idx]
    
    if feature in df_valid.columns:
        valid_pairs = df_valid[['blood_glucose', feature]].dropna()
        
        if len(valid_pairs) > 0:
            ax.scatter(valid_pairs[feature], valid_pairs['blood_glucose'], 
                      alpha=0.3, s=10, color='blue')
            
            # Add trend line
            if len(valid_pairs) > 1:
                z = np.polyfit(valid_pairs[feature], valid_pairs['blood_glucose'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid_pairs[feature].min(), valid_pairs[feature].max(), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
            corr = correlations.get(feature, 0)
            p_val = p_values.get(feature, 1)
            
            # Highlight new features
            title_color = 'green' if feature in new_features else 'black'
            
            ax.set_xlabel(feature, fontsize=9)
            ax.set_ylabel('Blood Glucose (mg/dL)', fontsize=9)
            ax.set_title(f'r={corr:.3f}, p={p_val:.4f}', fontsize=8, color=title_color)
            ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(len(available_features), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.tight_layout()
plt.savefig('glucose_correlation_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'glucose_correlation_analysis.png'")

# Feature statistics
print("\n" + "="*70)
print("FEATURE STATISTICS")
print("="*70)

for feature in available_features:
    if feature in df_valid.columns:
        feature_data = df_valid[feature].dropna()
        if len(feature_data) > 0:
            print(f"\n{feature}:")
            print(f"  Count:  {len(feature_data)}")
            print(f"  Mean:   {feature_data.mean():.4f}")
            print(f"  Std:    {feature_data.std():.4f}")
            print(f"  Min:    {feature_data.min():.4f}")
            print(f"  Max:    {feature_data.max():.4f}")

# Quality check for image_quality_score if it exists
if 'image_quality_score' in df_valid.columns:
    print("\n" + "="*70)
    print("IMAGE QUALITY ANALYSIS")
    print("="*70)
    quality_scores = df_valid['image_quality_score'].dropna()
    if len(quality_scores) > 0:
        low_quality = quality_scores[quality_scores < 30]
        print(f"Average quality score: {quality_scores.mean():.1f}/100")
        print(f"Low quality images (<30): {len(low_quality)} ({len(low_quality)/len(quality_scores)*100:.1f}%)")
        if len(low_quality) > 0:
            print("Consider retaking low quality images for better results")

# Check for duplicate features in old data
print("\n" + "="*70)
print("DATA MIGRATION CHECK")
print("="*70)

if 'pupil_dilation_rate' in df.columns and 'pupil_size' in df.columns:
    valid_both = df[['pupil_size', 'pupil_dilation_rate']].dropna()
    if len(valid_both) > 0:
        pupil_corr = valid_both.corr().iloc[0, 1]
        print(f"\nCorrelation between pupil_size and pupil_dilation_rate: {pupil_corr:.6f}")
        if abs(pupil_corr) > 0.99:
            print("⚠️  These features are duplicates! pupil_dilation_rate has been removed in new version.")
else:
    print("✓ No duplicate pupil features found (using updated schema)")

if 'pupil_response_time' in df.columns:
    print("⚠️  Found old 'pupil_response_time' column - renamed to 'capture_duration' in new version")
elif 'capture_duration' in df.columns:
    print("✓ Using updated 'capture_duration' column name")

plt.show()