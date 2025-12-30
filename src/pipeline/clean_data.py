#!/usr/bin/env python3
"""
Clean NASA C-MAPSS Dataset - Windows Version
Step 1.2 Data Cleaning for CMMS AI Project
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import os
import typing as t
from config import config

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs standard data cleaning and preprocessing steps on a CMMS-style dataset.
    This includes handling missing values, removing 3-sigma outliers using the 
    custom `remove_outliers_3sigma` function, and normalizing sensor readings.

    The function assumes sensor columns start with 'sensor_'.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing raw sensor and operational data.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with clean, outlier-free, and normalized sensor data.
    """

    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    if not any(col.startswith('sensor_') for col in df.columns):
        raise ValueError("DataFrame must contain sensor columns")
    
    # 1. Handling Missing Values
    print("\n--- 1. Missing Value Check & Removal ---")
    
    # Check and print the count and percentage of null values per column
    missing_counts: pd.Series = df.isnull().sum()
    missing_percentages: pd.Series = (missing_counts / len(df)) * 100

    print("Missing values count per column:")
    print(missing_counts)
    print("\nMissing values percentage per column:")
    print(missing_percentages)

    # Drop rows with any missing values and save to a new DataFrame
    df_cleaned: pd.DataFrame = df.dropna()

    print(f"\nOriginal rows: {len(df)}")
    print(f"Rows after dropping NaNs: {len(df_cleaned)}")

    # 2. Handling Outliers (3-Sigma Rule)
    print("\n--- 2. Outlier Removal (3-Sigma) ---")

    # Identify all columns containing sensor readings
    sensor_cols: t.List[str] = [col for col in df_cleaned.columns if col.startswith('sensor_')]

    # Remove rows where any sensor reading is > 3 standard deviations from the mean
    # The 'remove_outliers_3sigma' function handles the logic for Z-scores and skips constant features.
    df_no_outliers: pd.DataFrame = remove_outliers_3sigma_vectorized(df_cleaned, sensor_cols)
    
    print(f"Rows after dropping outliers: {len(df_no_outliers)}")

    # 3. Normalization (Min-Max Scaling)
    print("\n--- 3. Min-Max Normalization ---")

    # Initialize the MinMaxScaler from scikit-learn
    scaler: MinMaxScaler = MinMaxScaler()
    
    # Filter the list of sensor columns to only include those that were not skipped
    # This prevents the MinMaxScaler from failing due to zero variance (min=max)
    cols_to_scale: t.List[str] = [col for col in sensor_cols if df_no_outliers[col].std() > 1e-10]

    # Fit the scaler on the sensor data and transform the data to 0-1 scale
    df_no_outliers[cols_to_scale] = scaler.fit_transform(df_no_outliers[cols_to_scale])
    
    # 4. Final Data Quality Report
    print("\n--- 4. Data Quality Report for Step 1.2 ---")
    
    # Check missing values
    final_missing_percent: float = (df_no_outliers.isnull().sum().sum() / (len(df_no_outliers) * len(df_no_outliers.columns))) * 100

    # Check normalization only on scaled columns
    # Handle edge case where no columns need scaling
    if cols_to_scale:
        min_values: pd.Series = df_no_outliers[cols_to_scale].min()
        max_values: pd.Series = df_no_outliers[cols_to_scale].max()
        # Use tolerance for floating point comparison
        is_normalized: bool = (min_values.min() >= -1e-10) and (max_values.max() <= 1 + 1e-10)
        min_val_display = min_values.min()
        max_val_display = max_values.max()
    else:
        is_normalized = True  # No variable columns to scale
        min_val_display = 0.0
        max_val_display = 0.0

    print(f"Total Percentage of Missing Values in Final Data: {final_missing_percent:.4f}%")
    print(f"Target (<2% missing): {'✅ MET' if final_missing_percent < 2 else '❌ NOT MET'}")

    print("\nNormalization Check:")
    print(f"All *variable* sensor columns normalized to 0-1 scale: {'✅ MET' if is_normalized else '❌ NOT MET'}")
    print(f"Min value of scaled sensors: {min_val_display:.4f}")
    print(f"Max value of scaled sensors: {max_val_display:.4f}")
    print(f"Final dataset size: {len(df_no_outliers)} records")
    
    return df_no_outliers


def remove_outliers_3sigma_vectorized(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Vectorized outlier removal - much faster for many columns."""
    df_out = df.copy()
    
    # Filter columns with variance
    variable_cols = [col for col in columns if df_out[col].std() > 1e-10]
    
    if not variable_cols:
        return df_out
    
    # Vectorized Z-score calculation for all columns at once
    z_scores = np.abs((df_out[variable_cols] - df_out[variable_cols].mean()) / df_out[variable_cols].std())
    
    # Create mask: keep rows where ALL z-scores < 3
    mask = (z_scores < 3).all(axis=1)
    
    outliers_removed = (~mask).sum()
    print(f"\n[SUMMARY] Outliers removed: {outliers_removed} rows ({outliers_removed/len(df_out)*100:.2f}%)")
    
    return df_out[mask]

def main() -> None:
    """
    Main execution function for Step 1.2 Data Cleaning.
    Loads the raw dataset, calls the data cleaning pipeline, and saves the 
    processed DataFrame for use in the next step (Feature Engineering).
    """
    
    # --- Configuration ---
    # Define input and output paths using centralized config
    INPUT_DIR: Path = config.paths.raw_data
    OUTPUT_DIR: Path = config.paths.processed_data

    # Get all train files
    train_files: t.List[str] = [f for f in os.listdir(INPUT_DIR) if f.startswith('train_FD') and f.endswith('.txt')]

    if not train_files:
        print(f"❌ ERROR: No training files found in {INPUT_DIR}")
        return

    print(f"Found {len(train_files)} training file(s) to process:")
    for f in sorted(train_files):
        print(f"  📄 {f}")
    
    # Define the column names for C-MAPSS FD001 dataset
    COLUMN_NAMES: t.List[str] = [
        'unit_id', 'time_cycles', 
        'setting_1', 'setting_2', 'setting_3',
        'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6', 
        'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12', 
        'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16', 'sensor_17', 'sensor_18', 
        'sensor_19', 'sensor_20', 'sensor_21'
    ]

    print("\n--- Starting Step 1.2: Data Cleaning and Normalization ---")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each file
    for train_file in sorted(train_files):
        print("\n" + "="*70)
        print(f"PROCESSING: {train_file}")
        print("="*70)
        
        INPUT_FILE: str = os.path.join(INPUT_DIR, train_file)
        OUTPUT_FILE: str = os.path.join(OUTPUT_DIR, train_file.replace('.txt', '_cleaned.csv'))
        
        # --- 1. Load Data ---
        try:
            df_raw: pd.DataFrame = pd.read_csv(INPUT_FILE, sep=r'\s+', header=None, names=COLUMN_NAMES)
            print(f"Successfully loaded {train_file} with {len(df_raw)} records.")
        except FileNotFoundError:
            print(f"❌ ERROR: Input file '{INPUT_FILE}' not found.")
            continue
        except Exception as e:
            print(f"❌ An unexpected error occurred during file loading: {e}")
            continue

        # --- 2. Clean and Normalize Data ---
        df_clean_norm: pd.DataFrame = clean_dataset(df_raw)

        # --- 3. Save Cleaned Data ---
        print("\n--- Saving Cleaned Data ---")
        
        try:
            df_clean_norm.to_csv(OUTPUT_FILE, index=False)
            print(f"✅ Success: Cleaned data saved to {OUTPUT_FILE}")
        except Exception as e:
            print(f"❌ An error occurred while saving the file: {e}")

    print("\n" + "="*70)
    print("✅ ALL FILES PROCESSED")
    print("="*70)
    print("Ready for Step 1.3: Feature Engineering and Train/Test Split.")
    print("--- Step 1.2 Complete ---")
    

if __name__ == "__main__":
    main()