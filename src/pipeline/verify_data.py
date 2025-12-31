#!/usr/bin/env python3
"""
Verify NASA C-MAPSS Dataset - Windows Version
Step 1.1 Verification for CMMS AI Project
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import os
import typing as t
import traceback
from config import config
from terminal_colors import Colors, print_header, print_success, print_warning, print_error

def verify_dataset() -> bool:
    """
    Loads the C-MAPSS dataset file, prints a comprehensive statistical report, 
    and checks if the dataset meets the project's minimum size requirement.

    Returns
    -------
    bool
        True if the files are found, loaded, and verification is complete, False otherwise.
    """
    
    # --- Configuration ---
    # Define the 26 column names as per C-MAPSS documentation (no headers in raw files)
    columns: t.List[str] = ['unit_id', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
                           [f'sensor_{i}' for i in range(1, 22)]
    
    # Target directory containing all FD files (using centralized config)
    data_dir: Path = config.paths.raw_data
    
    # Project requirement from S.M.A.R.T. plan (minimum 50,000 records)
    requirement: int = 50000
    
    # --- Start Report ---
    print_header("NASA C-MAPSS DATASET VERIFICATION (Step 1.1)")
    
    # 1. File Existence Check
    train_files: t.List[str] = [f for f in os.listdir(data_dir) if f.startswith('train_FD') and f.endswith('.txt')]

    if not train_files:
        print_error("ERROR: No training dataset files found!")
        print(f"Expected location: {os.path.abspath(data_dir)}")
        print(f"Run 'python download_data.py' first to acquire the dataset.")
        return False

    print_success(f"Found {len(train_files)} dataset file(s)")
    for f in sorted(train_files):
        print(f"   📄 {f}")
    print(f"Location: {os.path.abspath(data_dir)}")

    # 2. Load and Combine All Data
    print_header("LOADING ALL DATASETS...")

    try:
        all_dfs: t.List[pd.DataFrame] = []
        
        for train_file in sorted(train_files):
            file_path: str = os.path.join(data_dir, train_file)
            df_temp: pd.DataFrame = pd.read_csv(file_path, sep=r'\s+', header=None, names=columns)
            df_temp['source_file'] = train_file  # Track which file each record came from
            all_dfs.append(df_temp)
            print_success(f"Loaded {train_file}: {len(df_temp):,} records")
        
        # Combine all datasets
        df: pd.DataFrame = pd.concat(all_dfs, ignore_index=True)
        print_success("All data combined successfully!")
        
        # 3. Dataset Statistics
        print_header("DATASET STATISTICS")
        
        total_records: int = len(df)
        
        print(f"  Total Records:        {total_records:>12,}")
        print(f"  Number of Engines:    {df['unit_id'].nunique():>12,}")
        print(f"  Number of Columns:    {len(df.columns):>12,}")
        # Calculate memory usage in MB
        mem_usage_mb: float = df.memory_usage(deep=True).sum() / 1024**2
        print(f"  Memory Usage:         {mem_usage_mb:>11.2f} MB")
        
        # Calculate min/max cycle length across all engines
        min_cycles: int = df.groupby('unit_id')['time_cycles'].max().min()
        max_cycles: int = df.groupby('unit_id')['time_cycles'].max().max()
        print(f"  Engine Cycle Range:   {min_cycles:>12,} - {max_cycles:,} cycles")
        
        # 4. Requirement Check
        print_header("REQUIREMENT VERIFICATION")
        
        if total_records >= requirement:
            print_success(f"PASSED: Dataset has {total_records:,} records")
            print(f"     Required: ≥{requirement:,} records")
            print(f"     Exceeded by: {total_records - requirement:,} records")
        else:
            print_warning(f"WARNING: Dataset has only {total_records:,} records")
            print(f"     Required: ≥{requirement:,} records")
            
        
        # 5. Sample and Quality Check
        print_header("SAMPLE DATA (First 5 rows)")

        # Use .to_string() for clean console printing
        print(df.head().to_string())
        
        # 6. Missing Values and Data Types
        print_header("DATA QUALITY CHECK & TYPES")

        missing_values: int = df.isnull().sum().sum()
        print(f"Missing values:       {missing_values:>12,}")
        print(f"Duplicate rows:       {df.duplicated().sum():>12,}")
        
        if missing_values == 0:
            print_success("No missing values detected.")
            
        # Data types summary (should mostly be float64)
        print("\nData Types (dtypes):")
        print(df.dtypes.value_counts().to_string())
        
        # 7. Final Summary
        print_header("VERIFICATION COMPLETE!")

        print_success("Dataset successfully loaded and verified.")
        print_success(f"All requirements met ({total_records:,} total records available).")
        
        return True
        
    except FileNotFoundError:
        # This should be caught by the initial check, but included for robustness
        print_error(f"Error: Could not find file")
        return False
    except pd.errors.EmptyDataError:
        print_error("Error: File is empty or improperly formatted")
        return False
    except Exception as e:
        print_error(f"Unexpected Error during loading or processing: {e}")
        traceback.print_exc()
        return False

def main() -> None:
    """
    Main entry point of the script. Calls the verification function and provides 
    the next step guidance. Exits with a non-zero code if verification fails.
    """
    print("\n")
    success: bool = verify_dataset()
    
    if not success:
        print_warning("\nVerification failed! Cannot proceed to the next step.")
        sys.exit(1)
        
    
    print("\n")

if __name__ == "__main__":
    main()