import pandas as pd
import os

# Read all three CSV files
print("Reading CSV files...")
labels1 = pd.read_csv('eye_glucose_data/labels1.csv')
labels2 = pd.read_csv('eye_glucose_data/labels2.csv')
labels_new = pd.read_csv('eye_glucose_data/labels.csv')

print(f"✓ labels1.csv: {len(labels1)} rows")
print(f"✓ labels2.csv: {len(labels2)} rows")
print(f"✓ labels.csv (new): {len(labels_new)} rows")

# Process labels1 and labels2 to new format
def convert_old_to_new_format(df_old):
    """Convert old CSV format to new format with updated columns."""
    df_new = pd.DataFrame()
    
    # Copy over compatible columns
    df_new['filename'] = df_old['filename']
    df_new['blood_glucose'] = df_old['blood_glucose']
    df_new['pupil_size'] = df_old['pupil_size']
    df_new['sclera_redness'] = df_old['sclera_redness']
    df_new['vein_prominence'] = df_old['vein_prominence']
    
    # Rename pupil_response_time to capture_duration
    df_new['capture_duration'] = df_old['pupil_response_time']
    
    df_new['ir_intensity'] = df_old['ir_intensity']
    df_new['scleral_vein_density'] = df_old['scleral_vein_density']
    df_new['ir_temperature'] = df_old['ir_temperature']
    df_new['tear_film_reflectivity'] = df_old['tear_film_reflectivity']
    df_new['sclera_color_balance'] = df_old['sclera_color_balance']
    df_new['vein_pulsation_intensity'] = df_old['vein_pulsation_intensity']
    df_new['birefringence_index'] = df_old['birefringence_index']
    
    # Add new columns as NaN (will be empty for old data)
    df_new['lens_clarity_score'] = None
    df_new['sclera_yellowness'] = None
    df_new['vessel_tortuosity'] = None
    df_new['image_quality_score'] = None
    
    # Note: pupil_dilation_rate is intentionally dropped (was duplicate)
    
    return df_new

print("\nConverting old data to new format...")
labels1_converted = convert_old_to_new_format(labels1)
labels2_converted = convert_old_to_new_format(labels2)

# Combine all three datasets
print("Merging all datasets...")
combined = pd.concat([labels1_converted, labels2_converted, labels_new], ignore_index=True)

print(f"\n✓ Total combined rows: {len(combined)}")

# Remove any duplicate filenames (keep the most recent one)
print("\nChecking for duplicates...")
duplicates = combined[combined.duplicated(subset=['filename'], keep=False)]
if len(duplicates) > 0:
    print(f"⚠️  Found {len(duplicates)} duplicate filenames")
    print("Keeping the last occurrence of each duplicate...")
    combined = combined.drop_duplicates(subset=['filename'], keep='last')
    print(f"✓ Removed duplicates. Now have {len(combined)} unique rows")
else:
    print("✓ No duplicates found")

# Show some statistics
print("\n" + "="*70)
print("DATASET STATISTICS")
print("="*70)

total_glucose = combined['blood_glucose'].notna() & (combined['blood_glucose'] != '')
print(f"Total samples: {len(combined)}")
print(f"Samples with glucose values: {total_glucose.sum()}")
print(f"Samples missing glucose: {len(combined) - total_glucose.sum()}")

# Count how many have new features
new_features_count = combined['lens_clarity_score'].notna().sum()
print(f"\nSamples with new features: {new_features_count}")
print(f"Samples with old features only: {len(combined) - new_features_count}")

# Save the backup
if os.path.exists('eye_glucose_data/labels.csv'):
    import shutil
    shutil.copy('eye_glucose_data/labels.csv', 'eye_glucose_data/labels_before_merge.csv')
    print("\n✓ Created backup: labels_before_merge.csv")

# Save the combined dataset
combined.to_csv('eye_glucose_data/labels_combined.csv', index=False, float_format='%.10f')
print("✓ Saved combined dataset to: labels_combined.csv")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("1. Review labels_combined.csv to make sure it looks correct")
print("2. If it looks good, rename it to labels.csv:")
print("   mv eye_glucose_data/labels_combined.csv eye_glucose_data/labels.csv")
print("3. Run analysis.py to see correlations across all your data!")
print("\nNote: Old samples won't have the new features (lens_clarity_score,")
print("sclera_yellowness, vessel_tortuosity, image_quality_score).")
print("These will be NaN/empty for historical data.")