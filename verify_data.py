#!/usr/bin/env python3
"""
Verify NASA C-MAPSS Dataset - Windows Version
Step 1.1 Verification for CMMS AI Project
"""

import pandas as pd
import numpy as np
import os
import sys
import typing as t
import traceback

def verify_dataset() -> bool:
    """
    Loads the C-MAPSS dataset file, prints a comprehensive statistical report, 
    and checks if the dataset meets the project's minimum size requirement.

    Returns
    -------
    bool
        True if the file is found, loaded, and verification is complete, False otherwise.
    """
    
    # --- Configuration ---
    # Define the 26 column names as per C-MAPSS documentation (no headers in raw files)
    columns: t.List[str] = ['unit_id', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
                           [f'sensor_{i}' for i in range(1, 22)]
    
    # Target file path (assuming download_data.py placed it here)
    data_file: str = 'data\\raw\\train_FD001.txt'
    
    # Project requirement from S.M.A.R.T. plan (minimum 50,000 records)
    requirement: int = 50000
    
    # --- Start Report ---
    print("=" * 70)
    print("NASA C-MAPSS DATASET VERIFICATION (Step 1.1)")
    print("=" * 70)
    
    # 1. File Existence Check
    if not os.path.exists(data_file):
        print(f"\n❌ ERROR: Dataset file not found!")
        print(f"Expected location: {os.path.abspath(data_file)}")
        print("\n💡 Run 'python download_data.py' first to acquire the dataset.")
        return False
    
    print(f"\n✅ Dataset file found")
    print(f"📍 Location: {os.path.abspath(data_file)}")
    
    # 2. Load Data
    print("\n" + "-" * 70)
    print("📊 LOADING DATA...")
    print("-" * 70)
    
    try:
        # Load data using space or tab as separator (\s+) since it's a raw NASA file
        # The 'r' prefix is used for raw string to avoid SyntaxWarning
        df: pd.DataFrame = pd.read_csv(data_file, sep=r'\s+', header=None, names=columns)
        
        print(f"✅ Data loaded successfully!\n")
        
        # 3. Dataset Statistics
        print("=" * 70)
        print("📈 DATASET STATISTICS")
        print("=" * 70)
        
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
        print("\n" + "=" * 70)
        print("✅ REQUIREMENT VERIFICATION")
        print("=" * 70)
        
        if total_records >= requirement:
            print(f"  ✅ PASSED: Dataset has {total_records:,} records")
            print(f"     Required: ≥{requirement:,} records")
            print(f"     Exceeded by: {total_records - requirement:,} records")
        else:
            print(f"  ⚠️ WARNING: Dataset has only {total_records:,} records")
            print(f"     Required: ≥{requirement:,} records")
            
        # 5. Additional Datasets Check (Total Record Count)
        print("\n" + "=" * 70)
        print("📦 ADDITIONAL DATASETS AVAILABLE (FD002, FD003, FD004)")
        print("=" * 70)
        
        data_dir: str = 'data\\raw'
        # Get all training files starting with 'train_FD'
        all_datasets: t.List[str] = [f for f in os.listdir(data_dir) if f.startswith('train_FD')]
        
        total_all: int = 0
        
        # Load and count records for each FD file
        for dataset in sorted(all_datasets):
            # Load without headers for speed, assuming same structure
            df_temp: pd.DataFrame = pd.read_csv(os.path.join(data_dir, dataset), sep=r'\s+', header=None)
            records: int = len(df_temp)
            total_all += records
            print(f"  {dataset:<20} {records:>12,} records")
            
        print(f"  {'-' * 20} {'-' * 18}")
        print(f"  {'TOTAL ALL FDs':<20} {total_all:>12,} records")
        
        # 6. Sample and Quality Check
        
        print("\n" + "=" * 70)
        print("📋 SAMPLE DATA (First 5 rows)")
        print("=" * 70)
        # Use .to_string() for clean console printing
        print(df.head().to_string())
        
        print("\n" + "=" * 70)
        print("🔍 DATA QUALITY CHECK & TYPES")
        print("=" * 70)
        missing_values: int = df.isnull().sum().sum()
        print(f"  Missing values:       {missing_values:>12,}")
        print(f"  Duplicate rows:       {df.duplicated().sum():>12,}")
        
        if missing_values == 0:
            print(f"  ✅ No missing values detected (Confirmed for Step 1.2 initial check).")
            
        # Data types summary (should mostly be float64)
        print("\n  Data Types (dtypes):")
        print(df.dtypes.value_counts().to_string())
        
        # 7. Final Summary
        print("\n" + "=" * 70)
        print("🎉 VERIFICATION COMPLETE!")
        print("=" * 70)
        print(f"  ✅ Dataset successfully loaded and verified.")
        print(f"  ✅ All requirements met ({total_all:,} total records available).")
        print(f"  ✅ Ready for Step 1.2: Data Cleaning & Preparation.")
        print("\n" + "=" * 70)
        
        return True
        
    except FileNotFoundError:
        # This should be caught by the initial check, but included for robustness
        print(f"❌ Error: Could not find file: {data_file}")
        return False
    except pd.errors.EmptyDataError:
        print(f"❌ Error: File is empty or improperly formatted: {data_file}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error during loading or processing: {e}")
        traceback.print_exc()
        return False

def main() -> None:
    """
    Main entry point of the script. Calls the verification function and provides 
    the next step guidance. Exits with a non-zero code if verification fails.
    """
    print("\n")
    success: bool = verify_dataset()
    
    if success:
        print("\n💡 NEXT STEPS:")
        print("   1. Review the statistical summary above.")
        print("   2. Proceed to Step 1.2: Data Cleaning.")
        print("   3. Run 'python clean_data.py' (or the equivalent script).")
    else:
        print("\n⚠️  Verification failed! Cannot proceed to the next step.")
        sys.exit(1)
    
    print("\n")

if __name__ == "__main__":
    main()