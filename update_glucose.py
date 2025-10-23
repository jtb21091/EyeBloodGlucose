#!/usr/bin/env python3
"""
Helper script to quickly update blood glucose values in the CSV file.
Usage:
    python update_glucose.py 120        # Set ALL empty entries to 120 mg/dL
    python update_glucose.py --latest 95   # Set only the most recent empty entries to 95 mg/dL
    python update_glucose.py --interactive  # Interactive mode to set individual values
    python update_glucose.py --session session_20240101_120000 105  # Fill a single batch session
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

LABELS_FILE = "eye_glucose_data/labels.csv"

CSV_COLUMNS = [
    "filename",
    "session_id",
    "blood_glucose",
    "pupil_size",
    "sclera_redness",
    "vein_prominence",
    "capture_duration",
    "ir_intensity",
    "scleral_vein_density",
    "ir_temperature",
    "tear_film_reflectivity",
    "sclera_color_balance",
    "vein_pulsation_intensity",
    "birefringence_index",
    "lens_clarity_score",
    "sclera_yellowness",
    "vessel_tortuosity",
    "image_quality_score"
]


def ensure_columns(df):
    """Ensure expected columns exist and align with capture outputs."""
    changed = False

    for col in CSV_COLUMNS:
        if col not in df.columns:
            if col in ("filename", "session_id"):
                df[col] = ""
            elif col == "blood_glucose":
                df[col] = np.nan
            else:
                df[col] = np.nan
            changed = True

    desired_order = [col for col in CSV_COLUMNS if col in df.columns]
    extra_columns = [col for col in df.columns if col not in CSV_COLUMNS]
    ordered_columns = desired_order + extra_columns

    if list(df.columns) != ordered_columns:
        df = df[ordered_columns]
        changed = True

    return df, changed

def load_data():
    """Load the labels CSV file"""
    if not os.path.exists(LABELS_FILE):
        print(f"❌ CSV file not found: {LABELS_FILE}")
        return None
    df = pd.read_csv(LABELS_FILE)
    df, changed = ensure_columns(df)
    if changed:
        df.to_csv(LABELS_FILE, index=False, float_format='%.10f')
    return df

def update_all_empty(df, glucose_value):
    """Update all empty blood glucose entries"""
    empty_mask = (df['blood_glucose'] == '') | df['blood_glucose'].isna()
    count = empty_mask.sum()
    
    if count == 0:
        print("✅ No empty blood glucose entries found.")
        return df, 0
    
    df.loc[empty_mask, 'blood_glucose'] = glucose_value
    print(f"✅ Updated {count} empty entries to {glucose_value} mg/dL")
    return df, count

def update_latest_empty(df, glucose_value, count=None):
    """Update the most recent empty blood glucose entries"""
    empty_mask = (df['blood_glucose'] == '') | df['blood_glucose'].isna()
    empty_indices = df[empty_mask].index
    
    if len(empty_indices) == 0:
        print("✅ No empty blood glucose entries found.")
        return df, 0
    
    if count is None:
        count = len(empty_indices)
    
    # Take the last 'count' empty entries (most recent)
    latest_empty = empty_indices[-count:] if count <= len(empty_indices) else empty_indices
    
    df.loc[latest_empty, 'blood_glucose'] = glucose_value
    print(f"✅ Updated {len(latest_empty)} most recent empty entries to {glucose_value} mg/dL")
    
    # Show which files were updated
    updated_files = df.loc[latest_empty, 'filename'].tolist()
    for f in updated_files:
        print(f"   📄 {f}")
    
    return df, len(latest_empty)


def update_session(df, session_id, glucose_value, include_filled=False):
    """Update all entries for a particular capture session."""
    if 'session_id' not in df.columns:
        print("⚠️ No session_id column found in CSV. Nothing to update.")
        return df, 0

    session_mask = df['session_id'] == session_id
    if not session_mask.any():
        print(f"❌ Session not found: {session_id}")
        return df, 0

    if include_filled:
        target_mask = session_mask
    else:
        empty_mask = (df['blood_glucose'] == '') | df['blood_glucose'].isna()
        target_mask = session_mask & empty_mask

    target_indices = df[target_mask].index

    if len(target_indices) == 0:
        if include_filled:
            print("✅ All entries already set for this session.")
        else:
            print("✅ No empty entries remaining for this session.")
        return df, 0

    df.loc[target_indices, 'blood_glucose'] = glucose_value
    print(f"✅ Updated {len(target_indices)} entries to {glucose_value} mg/dL for session {session_id}")

    updated_files = df.loc[target_indices, 'filename'].tolist()
    for f in updated_files:
        print(f"   📄 {f}")

    return df, len(target_indices)

def interactive_mode(df):
    """Interactive mode to set individual blood glucose values"""
    empty_mask = (df['blood_glucose'] == '') | df['blood_glucose'].isna()
    empty_entries = df[empty_mask]
    
    if len(empty_entries) == 0:
        print("✅ No empty blood glucose entries found.")
        return df, 0
    
    print(f"\n📋 Found {len(empty_entries)} empty entries:")
    print("=" * 80)
    
    updated_count = 0
    for idx, row in empty_entries.iterrows():
        filename = row['filename']
        timestamp = filename.split('_')[1:3] if '_' in filename else ['unknown', 'time']
        
        print(f"\n📄 File: {filename}")
        print(f"🕐 Time: {timestamp[0]}_{timestamp[1] if len(timestamp) > 1 else ''}")
        print(f"👁️ Quality: {row.get('image_quality_score', 'N/A'):.1f}/100" if pd.notna(row.get('image_quality_score')) else "👁️ Quality: N/A")
        
        while True:
            try:
                glucose_input = input("🩸 Enter blood glucose (mg/dL) or 's' to skip, 'q' to quit: ").strip()
                
                if glucose_input.lower() == 'q':
                    print("🛑 Quitting interactive mode.")
                    return df, updated_count
                elif glucose_input.lower() == 's':
                    print("⏭️ Skipped.")
                    break
                else:
                    glucose_value = float(glucose_input)
                    if 20 <= glucose_value <= 600:  # Reasonable glucose range
                        df.loc[idx, 'blood_glucose'] = glucose_value
                        updated_count += 1
                        print(f"✅ Set to {glucose_value} mg/dL")
                        break
                    else:
                        print("⚠️ Please enter a value between 20-600 mg/dL")
            except ValueError:
                print("⚠️ Please enter a valid number, 's' to skip, or 'q' to quit")
    
    return df, updated_count

def show_status(df):
    """Show current status of the dataset"""
    total_entries = len(df)
    empty_count = ((df['blood_glucose'] == '') | df['blood_glucose'].isna()).sum()
    filled_count = total_entries - empty_count
    
    print(f"\n📊 DATASET STATUS:")
    print("=" * 40)
    print(f"📄 Total entries: {total_entries}")
    print(f"✅ Filled entries: {filled_count}")
    print(f"⭕ Empty entries: {empty_count}")
    print(f"📈 Completion: {(filled_count/total_entries*100):.1f}%" if total_entries > 0 else "📈 Completion: 0%")
    
    if empty_count > 0:
        print(f"\n📋 Recent empty entries:")
        empty_mask = (df['blood_glucose'] == '') | df['blood_glucose'].isna()
        recent_empty = df[empty_mask].tail(5)
        for _, row in recent_empty.iterrows():
            print(f"   📄 {row['filename']}")

    if 'session_id' in df.columns:
        print(f"\n🆔 Sessions with empty entries:")
        empty_mask = (df['blood_glucose'] == '') | df['blood_glucose'].isna()
        session_counts = df[empty_mask].groupby('session_id').size()
        if session_counts.empty:
            print("   ✓ All sessions filled")
        else:
            for session_id, count in session_counts.sort_values(ascending=False).head(5).items():
                print(f"   {session_id}: {count} remaining")

def main():
    if len(sys.argv) == 1:
        # No arguments - show status
        df = load_data()
        if df is not None:
            show_status(df)
            print(f"\n💡 Usage:")
            print(f"  python update_glucose.py 120              # Update all empty to 120")
            print(f"  python update_glucose.py --latest 95 5    # Update latest 5 empty to 95")
            print(f"  python update_glucose.py --interactive    # Interactive mode")
            print(f"  python update_glucose.py --session <session_id> 110  # Update one session")
        return
    
    df = load_data()
    if df is None:
        return
    
    updated_count = 0
    
    if sys.argv[1] == "--interactive":
        df, updated_count = interactive_mode(df)
    elif sys.argv[1] == "--latest":
        if len(sys.argv) < 3:
            print("❌ Usage: python update_glucose.py --latest <glucose_value> [count]")
            return
        
        glucose_value = float(sys.argv[2])
        count = int(sys.argv[3]) if len(sys.argv) > 3 else None
        df, updated_count = update_latest_empty(df, glucose_value, count)
    elif sys.argv[1] == "--session":
        if len(sys.argv) < 4:
            print("❌ Usage: python update_glucose.py --session <session_id> <glucose_value> [--all]")
            return

        session_id = sys.argv[2]
        glucose_value = float(sys.argv[3])
        include_filled = len(sys.argv) > 4 and sys.argv[4] == "--all"
        df, updated_count = update_session(df, session_id, glucose_value, include_filled)
    else:
        # Simple mode - update all empty
        glucose_value = float(sys.argv[1])
        df, updated_count = update_all_empty(df, glucose_value)
    
    # Save if any updates were made
    if updated_count > 0:
        # Create backup
        backup_file = f"{LABELS_FILE}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        original_df = pd.read_csv(LABELS_FILE)
        original_df.to_csv(backup_file, index=False)
        print(f"💾 Backup created: {backup_file}")
        
        # Save updated file
        df.to_csv(LABELS_FILE, index=False, float_format='%.10f')
        print(f"💾 Updated CSV saved: {LABELS_FILE}")
        
        print(f"\n🎉 Summary: Updated {updated_count} entries")
    
    # Show final status
    show_status(df)

if __name__ == "__main__":
    main()