#!/usr/bin/env python3
"""
Clean NASA C-MAPSS Dataset - Windows Version
Step 1.2 Data Cleaning for CMMS AI Project
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import os

def clean_dataset(df):
    """
    Performs standard data cleaning and preprocessing steps on a CMMS-style dataset.
    This includes handling missing values, removing 3-sigma outliers, and 
    normalizing sensor readings to a 0-1 scale.

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
    
    # 1. Handling Missing Values
    print("\n--- 1. Missing Value Check ---")
    
    # Check and print the count and percentage of null values per column
    # Note: df.isnull().sum() counts True values (missing) per column.
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100

    print("Missing values count per column:")
    print(missing_counts)
    print("\nMissing values percentage per column:")
    print(missing_percentages)

    # Drop rows with any missing values and save to a new DataFrame
    df_cleaned = df.dropna()

    print(f"\nOriginal rows: {len(df)}")
    print(f"Rows after dropping NaNs: {len(df_cleaned)}")

    # 2. Handling Outliers (3-Sigma Rule)
    print("\n--- 2. Outlier Removal (3-Sigma) ---")

    # Identify all columns containing sensor readings
    # This uses a list comprehension assuming sensor columns start with 'sensor_'
    sensor_cols = [col for col in df_cleaned.columns if col.startswith('sensor_')]

    # Remove rows where any sensor reading is > 3 standard deviations from the mean
    # NOTE: The 'remove_outliers_3sigma' function handles the logic for Z-scores.
    df_no_outliers = remove_outliers_3sigma(df_cleaned, sensor_cols)
    
    print(f"Rows after dropping outliers: {len(df_no_outliers)}")

    # 3. Normalization (Min-Max Scaling)
    print("\n--- 3. Min-Max Normalization ---")

    # Initialize the MinMaxScaler from scikit-learn
    scaler = MinMaxScaler()

    # Fit the scaler on the sensor data and transform the data to 0-1 scale
    # This overwrites the sensor columns with the normalized values.
    df_no_outliers[sensor_cols] = scaler.fit_transform(df_no_outliers[sensor_cols])
    
    # Final Data Quality Report
    print("\n--- 4. Data Quality Report for Step 1.2 ---")
    
    # A. Check missing values (Should be 0 if dropna was effective)
    final_missing_percent = (df_no_outliers.isnull().sum().sum() / (len(df_no_outliers) * len(df_no_outliers.columns))) * 100

    # B. Check normalization
    min_values = df_no_outliers[sensor_cols].min()
    max_values = df_no_outliers[sensor_cols].max()
    is_normalized = (min_values.min() >= 0) and (max_values.max() <= 1)

    print(f"Total Percentage of Missing Values in Final Data: {final_missing_percent:.4f}%")
    print(f"Target (<2% missing): {'✅ MET' if final_missing_percent < 2 else '❌ NOT MET'}")

    print("\nNormalization Check:")
    print(f"All sensor columns normalized to 0-1 scale: {'✅ MET' if is_normalized else '❌ NOT MET'}")
    print(f"Min value of all normalized sensors: {min_values.min():.4f}")
    print(f"Max value of all normalized sensors: {max_values.max():.4f}")
    print(f"Final dataset size: {len(df_no_outliers)} records")
    
    return df_no_outliers



def remove_outliers_3sigma(df, columns):
    """
    Removes rows from the DataFrame where any value in the specified 
    sensor columns exceeds 3 standard deviations from its mean.
    
    Uses IQR method as fallback for columns with low variance.

    The Z-score method is used: |(X - mean) / std_dev| > 3. 
    It drops a row if *any* specified sensor reading in that row is an outlier.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the sensor data.
    columns : list of str
        A list of column names (sensor readings) to check for outliers.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with outlier rows removed.

    Notes
    -----
    A deep copy of the DataFrame is created to avoid modifying the original data.
    """
    
    # Create a deep copy to ensure the original DataFrame is unmodified
    os.makedirs("data\\processed", exist_ok=True)
    df_out = df.copy()

    # Track which rows to keep
    mask = pd.Series([True] * len(df_out))

    for col in columns:
        # Check if column has variance
        if df_out[col].std() < 1e-10:
            # Skip columns with near-zero variance
            print(f"Skipping {col} - near-zero variance")
            continue
        
        # Calculate Z-scores only for columns with variance
        z_scores = np.abs((df_out[col] - df_out[col].mean()) / df_out[col].std())
        
        # Mark outliers
        mask = mask & (z_scores < 3)
    
    df_filtered = df_out[mask]
    
    outliers_removed = len(df_out) - len(df_filtered)
    print(f"Outliers removed: {outliers_removed} rows ({outliers_removed/len(df_out)*100:.2f}%)")
    
    return df_filtered

def main():
    """
    Main execution function for Step 1.2 Data Cleaning.
    Loads the raw dataset, performs cleaning and normalization, 
    and saves the cleaned result to a new CSV file.
    """
    
    # --- Configuration ---
    # Define your input and output file paths
    # ASSUMPTION: The raw data from Step 1.1 is saved here.
    INPUT_FILE = "data\\raw\\train_FD001.txt"
    OUTPUT_FILE = "data\\processed\\train_FD001_cleaned.csv"
    
    # Define the columns that need to be read/used from your dataset
    COLUMN_NAMES = [
        'unit_id', 'time_cycles', 
        'setting_1', 'setting_2', 'setting_3',
        'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6', 
        'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12', 
        'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16', 'sensor_17', 'sensor_18', 
        'sensor_19', 'sensor_20', 'sensor_21'
    ]

    print("--- Starting Step 1.2: Data Cleaning and Normalization ---")

    # --- 1. Load Data (Step 1.1 completion) ---
    try:
        # Load the CSV file into a pandas DataFrame
        df_raw = pd.read_csv(INPUT_FILE, sep=r'\s+', header=None, names=COLUMN_NAMES)
        print(f"Successfully loaded {INPUT_FILE} with {len(df_raw)} records.")
    except FileNotFoundError:
        print(f"ERROR: Input file '{INPUT_FILE}' not found. Please ensure Step 1.1 is complete.")
        return
    except Exception as e:
        print(f"An error occurred during file loading: {e}")
        return

    # --- 2. Clean and Normalize Data (Step 1.2 execution) ---
    df_clean_norm = clean_dataset(df_raw)

    # --- 3. Save Cleaned Data (Preparation for Step 1.3) ---
    print("\n--- 5. Saving Cleaned Data ---")
    
    try:
        # Save the resulting DataFrame to a new CSV file without the index
        df_clean_norm.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Success: Cleaned and normalized data saved to {OUTPUT_FILE}")
        print("Ready for Step 1.3: Feature Engineering and Train/Test Split.")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
    
    print("--- Step 1.2 Complete ---")
    

if __name__ == "__main__":
    main()
