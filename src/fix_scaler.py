"""
Fix Scaler - Refit scaler on combined training data
Creates a single, consistent scaler for the entire pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import config
import joblib
import json

def main():
    print("="*70)
    print("FIXING SCALER - Creating consistent normalization")
    print("="*70)
    
    # Load your existing processed data
    print("\n📂 Loading processed datasets...")
    train_df = pd.read_csv(config.paths.train_file)
    val_df = pd.read_csv(config.paths.val_file)

    print(f"✅ Loaded train: {len(train_df)} records")
    print(f"✅ Loaded validation: {len(val_df)} records")
    
    # Identify and filter columns
    sensor_cols = [c for c in train_df.columns if c.startswith('sensor_')]
    cols_to_scale = [c for c in sensor_cols if train_df[c].std() > 1e-10]

    print(f"📊 Will scale {len(cols_to_scale)} columns (skipping constant sensors)")
    print(f"   Columns to scale: {cols_to_scale[:5]}... (showing first 5)")
    
    # Fit scaler on training data ONLY
    print("\n⚙️  Fitting scaler on TRAINING data only...")
    scaler = MinMaxScaler()
    train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
    
    # Transform validation data using the SAME scaler
    print("⚙️  Transforming VALIDATION data with fitted scaler...")
    val_df[cols_to_scale] = scaler.transform(val_df[cols_to_scale])
    
    # Verify normalization
    print("\n📈 Verification:")
    print(f"   Training min: {train_df[cols_to_scale].min().min():.4f}")
    print(f"   Training max: {train_df[cols_to_scale].max().max():.4f}")
    print(f"   Validation min: {val_df[cols_to_scale].min().min():.4f}")
    print(f"   Validation max: {val_df[cols_to_scale].max().max():.4f}")
    
    # Save the scaler
    scaler_path = config.paths.models_root / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"\n💾 Scaler saved to {scaler_path}")
    
    # Save column names
    columns_path = config.paths.models_root / 'scaler_columns.json'
    with open(columns_path, 'w') as f:
        json.dump(cols_to_scale, f, indent=2)
    print(f"💾 Column names saved to {columns_path}")
    
    # Overwrite your processed files - FIX THESE LINES:
    train_df.to_csv(config.paths.train_file, index=False)
    val_df.to_csv(config.paths.val_file, index=False)
    print(f"\n💾 Updated {config.paths.train_file}")
    print(f"💾 Updated {config.paths.val_file}")
    
    print("\n" + "="*70)
    print("✅ SCALER FIX COMPLETE!")
    print("="*70)
    print("\n📋 Next steps:")
    print("   1. Run your XGBoost training script (train_xgboost.py)")
    print("   2. Your model will now train on consistently-scaled data")
    print("   3. Use models/scaler.pkl for inference pipeline")

if __name__ == "__main__":
    main()