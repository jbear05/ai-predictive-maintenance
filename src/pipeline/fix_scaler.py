"""
Fix Scaler - Refit scaler on combined training data
Creates a single, consistent scaler for the entire pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import config
import joblib
import json
from terminal_colors import Colors, print_header, print_success, print_warning, print_error

def main():

    print_header(f"FIXING SCALER - Creating consistent normalization")
    
    # Load your existing processed data
    print(f"\nLoading processed datasets...")
    train_df = pd.read_csv(config.paths.train_file)
    val_df = pd.read_csv(config.paths.val_file)

    print_success(f"Loaded train: {len(train_df)} records")
    print_success(f"Loaded validation: {len(val_df)} records")
    
    # Identify and filter columns
    sensor_cols = [c for c in train_df.columns if c.startswith('sensor_')]
    cols_to_scale = [c for c in sensor_cols if train_df[c].std() > 1e-10]

    print(f"Will scale {len(cols_to_scale)} columns (skipping constant sensors)")
    print(f"Columns to scale: {cols_to_scale[:5]}... (showing first 5)")
    
    # Fit scaler on training data ONLY
    print(f"\nFitting scaler on TRAINING data only...")
    scaler = MinMaxScaler()
    train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
    
    # Transform validation data using the SAME scaler
    print(f"Transforming VALIDATION data with fitted scaler...")
    val_df[cols_to_scale] = scaler.transform(val_df[cols_to_scale])
    
    # Verify normalization
    print(f"\nVerification:")
    print(f"   Training min: {train_df[cols_to_scale].min().min():.4f}")
    print(f"   Training max: {train_df[cols_to_scale].max().max():.4f}")
    print(f"   Validation min: {val_df[cols_to_scale].min().min():.4f}")
    print(f"   Validation max: {val_df[cols_to_scale].max().max():.4f}")
    
    # Save the scaler
    scaler_path = config.paths.models_root / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print_success(f"Scaler saved to {scaler_path}")
    
    # Save column names
    columns_path = config.paths.models_root / 'scaler_columns.json'
    with open(columns_path, 'w') as f:
        json.dump(cols_to_scale, f, indent=2)
    print_success(f"Column names saved to {columns_path}")
    
    # Overwrite your processed files - FIX THESE LINES:
    train_df.to_csv(config.paths.train_file, index=False)
    val_df.to_csv(config.paths.val_file, index=False)
    print(f"\nUpdated {config.paths.train_file}")
    print(f"Updated {config.paths.val_file}")

if __name__ == "__main__":
    main()