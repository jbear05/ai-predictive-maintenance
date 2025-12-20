"""
Fix Scaler - Refit scaler on combined training data
Creates a single, consistent scaler for the entire pipeline
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import json
import os

def main():
    print("="*70)
    print("FIXING SCALER - Creating consistent normalization")
    print("="*70)
    
    # Load your existing processed data
    print("\n📂 Loading processed datasets...")
    train_df = pd.read_csv('data/processed/train_processed.csv')
    val_df = pd.read_csv('data/processed/val_processed.csv')
    print(f"✅ Loaded train: {len(train_df)} records")
    print(f"✅ Loaded validation: {len(val_df)} records")
    
    # Identify sensor columns
    sensor_cols = [col for col in train_df.columns if col.startswith('sensor_')]
    print(f"\n🔍 Found {len(sensor_cols)} sensor columns")
    
    # Only scale columns with variance
    cols_to_scale = [col for col in sensor_cols if train_df[col].std() > 1e-10]
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
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\n💾 Scaler saved to models/scaler.pkl")
    
    # Save column names
    with open('models/scaler_columns.json', 'w') as f:
        json.dump(cols_to_scale, f, indent=2)
    print("💾 Column names saved to models/scaler_columns.json")
    
    # Overwrite your processed files with properly scaled data
    train_df.to_csv('data/processed/train_processed.csv', index=False)
    val_df.to_csv('data/processed/val_processed.csv', index=False)
    print("\n💾 Updated data/processed/train_processed.csv")
    print("💾 Updated data/processed/val_processed.csv")
    
    print("\n" + "="*70)
    print("✅ SCALER FIX COMPLETE!")
    print("="*70)
    print("\n📋 Next steps:")
    print("   1. Run your XGBoost training script (train_xgboost.py)")
    print("   2. Your model will now train on consistently-scaled data")
    print("   3. Use models/scaler.pkl for inference pipeline")

if __name__ == "__main__":
    main()