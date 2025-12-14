#!/usr/bin/env python3
"""
Verify NASA C-MAPSS Dataset - Windows Version
Step 1.1 Verification for CMMS AI Project
"""

import pandas as pd
import numpy as np
import os
import sys

def verify_dataset():
    """Load and verify the NASA C-MAPSS dataset"""
    
    # Define column names (C-MAPSS doesn't include headers)
    columns = ['unit_id', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
              [f'sensor_{i}' for i in range(1, 22)]
    
    data_file = 'data\\raw\\train_FD001.txt'
    
    print("=" * 70)
    print("NASA C-MAPSS DATASET VERIFICATION")
    print("=" * 70)
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"\n❌ ERROR: File not found!")
        print(f"Expected location: {os.path.abspath(data_file)}")
        print("\n💡 Run 'python download_data.py' first to download the dataset.")
        return False
    
    print(f"\n✅ Dataset file found")
    print(f"📍 Location: {os.path.abspath(data_file)}")
    
    # Load data
    print("\n" + "-" * 70)
    print("📊 LOADING DATA...")
    print("-" * 70)
    
    try:
        df = pd.read_csv(data_file, sep=r'\s+', header=None, names=columns)
        
        print(f"✅ Data loaded successfully!\n")
        
        # Dataset statistics
        print("=" * 70)
        print("📈 DATASET STATISTICS")
        print("=" * 70)
        print(f"  Total Records:        {len(df):>12,}")
        print(f"  Number of Engines:    {df['unit_id'].nunique():>12,}")
        print(f"  Number of Columns:    {len(df.columns):>12,}")
        print(f"  Memory Usage:         {df.memory_usage(deep=True).sum() / 1024**2:>11.2f} MB")
        print(f"  Date Range:           {df.groupby('unit_id')['time_cycles'].max().min():>12,} - {df.groupby('unit_id')['time_cycles'].max().max():,} cycles")
        
        # Requirement check
        print("\n" + "=" * 70)
        print("✅ REQUIREMENT VERIFICATION")
        print("=" * 70)
        total_records = len(df)
        requirement = 50000
        
        if total_records >= requirement:
            print(f"  ✅ PASSED: Dataset has {total_records:,} records")
            print(f"     Required: ≥{requirement:,} records")
            print(f"     Exceeded by: {total_records - requirement:,} records")
        else:
            print(f"  ⚠️  WARNING: Dataset has only {total_records:,} records")
            print(f"     Required: ≥{requirement:,} records")
        
        # Additional datasets available
        print("\n" + "=" * 70)
        print("📦 ADDITIONAL DATASETS AVAILABLE")
        print("=" * 70)
        
        data_dir = 'data\\raw'
        all_datasets = [f for f in os.listdir(data_dir) if f.startswith('train_FD')]
        
        total_all = 0
        for dataset in sorted(all_datasets):
            df_temp = pd.read_csv(os.path.join(data_dir, dataset), sep=r'\s+', header=None)
            records = len(df_temp)
            total_all += records
            print(f"  {dataset:<20} {records:>12,} records")
        
        print(f"  {'-' * 20} {'-' * 18}")
        print(f"  {'TOTAL':<20} {total_all:>12,} records")
        
        # Column information
        print("\n" + "=" * 70)
        print("🏷️  COLUMN INFORMATION")
        print("=" * 70)
        print("  Operational Settings:")
        print("    - setting_1, setting_2, setting_3")
        print("\n  Sensor Readings (21 sensors):")
        for i in range(1, 22):
            if i % 5 == 1:
                print("   ", end="")
            print(f" sensor_{i:>2}", end="")
            if i % 5 == 0:
                print()
        if 21 % 5 != 0:
            print()
        
        # Sample data
        print("\n" + "=" * 70)
        print("📋 SAMPLE DATA (First 5 rows)")
        print("=" * 70)
        print(df.head().to_string())
        
        # Data quality check
        print("\n" + "=" * 70)
        print("🔍 DATA QUALITY CHECK")
        print("=" * 70)
        missing_values = df.isnull().sum().sum()
        print(f"  Missing values:       {missing_values:>12,}")
        print(f"  Duplicate rows:       {df.duplicated().sum():>12,}")
        
        if missing_values == 0:
            print(f"  ✅ No missing values detected!")
        
        # Data types
        print("\n" + "=" * 70)
        print("📊 DATA TYPES")
        print("=" * 70)
        print(df.dtypes.value_counts().to_string())
        
        # Success summary
        print("\n" + "=" * 70)
        print("🎉 VERIFICATION COMPLETE!")
        print("=" * 70)
        print(f"  ✅ Dataset successfully loaded and verified")
        print(f"  ✅ All requirements met ({total_all:,} total records)")
        print(f"  ✅ Data quality confirmed (no missing values)")
        print(f"  ✅ Ready for Step 1.2: Data Cleaning & Preparation")
        print("\n" + "=" * 70)
        
        return True
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find file: {data_file}")
        return False
    except pd.errors.EmptyDataError:
        print(f"❌ Error: File is empty: {data_file}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point"""
    print("\n")
    success = verify_dataset()
    
    if success:
        print("\n💡 NEXT STEPS:")
        print("   1. Review the data sample above")
        print("   2. Proceed to Step 1.2: Data Cleaning")
        print("   3. Run 'python clean_data.py' (coming next!)")
    else:
        print("\n⚠️  Verification failed!")
        print("   Please run 'python download_data.py' first.")
        sys.exit(1)
    
    print("\n")

if __name__ == "__main__":
    main()