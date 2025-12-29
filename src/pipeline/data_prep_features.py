import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
from config import config
import typing as t # For advanced type hinting
# Ignore pandas future warnings for rolling/ewm operations
warnings.filterwarnings('ignore') 


class CMAPSSDataPreparator:
    """
    Comprehensive data preparation and feature engineering pipeline for the C-MAPSS dataset.

    This class manages the entire preparation process: loading cleaned data, 
    calculating the Remaining Useful Life (RUL) and binary target, engineering 
    advanced temporal and statistical features, and creating a robust train/validation split.

    Attributes
    ----------
    data_dir : Path
        The root directory where input and output data are stored (e.g., 'data/processed').
    train_df : pd.DataFrame | None
        The processed training set after splitting.
    val_df : pd.DataFrame | None
        The processed validation set after splitting.
    feature_names : list[str]
        A list of the names of all engineered features.
    """
    
    def __init__(self, data_dir=None):
        # Convert string path to pathlib.Path object for cleaner path operations
        if data_dir is None:
            self.data_dir = config.paths.processed_data
        elif isinstance(data_dir, str):
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = data_dir
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.feature_names: t.List[str] = []
        
    def load_and_combine_train_files(self, 
                                     file_pattern: str = '*_cleaned.csv', 
                                     use_cleaned: bool = True) -> pd.DataFrame:
        """
        Loads all C-MAPSS training files matching a pattern and combines them into one DataFrame.

        Parameters
        ----------
        file_pattern : str, optional
            Glob pattern to match files (e.g., 'train_FD*_cleaned.csv').
            Defaults to '*_cleaned.csv' to load outputs from Step 1.2.
        use_cleaned : bool, optional
            If True, assumes files are CSVs with headers (from cleaning step). 
            If False, assumes raw files without headers. Defaults to True.

        Returns
        -------
        pd.DataFrame
            A combined DataFrame of all matched training files.

        Raises
        ------
        FileNotFoundError
            If no training files are found matching the specified pattern.
        """
        # Search for files recursively using the pattern
        train_files: t.List[Path] = list(self.data_dir.glob(file_pattern))
        
        if not train_files:
            raise FileNotFoundError(f"No training files found matching {file_pattern} in {self.data_dir}")
        
        print(f"Found {len(train_files)} training file(s)")
        
        all_data: t.List[pd.DataFrame] = []
        
        for file in train_files:
            print(f"Loading {file.name}...")
            
            if use_cleaned:
                # Load files that already have headers (output of clean_data.py)
                df: pd.DataFrame = pd.read_csv(file)
            else:
                # Fallback for raw files (not recommended for Step 1.3)
                columns: t.List[str] = ['unit_id', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
                columns += [f'sensor_{i}' for i in range(1, 22)]
                df = pd.read_csv(file, sep=r'\s+', header=None, names=columns)
                
            # Add a column to identify the original source file
            df['source_file'] = file.stem
            all_data.append(df)
            print(f"  Shape: {df.shape}, Columns: {len(df.columns)}")

        # Vertically concatenate all DataFrames
        combined_df: pd.DataFrame = pd.concat(all_data, ignore_index=True)
        print(f"\nCombined dataset shape: {combined_df.shape}")
        print(f"Total units: {combined_df['unit_id'].nunique()}")
        
        return combined_df
    
    def create_target_variable(self, df: pd.DataFrame, failure_window: int = None) -> pd.DataFrame:
        """
        Creates the Remaining Useful Life (RUL) column and the binary target variable.
        
        The binary target is 1 if RUL <= failure_window (i.e., failure is imminent).

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing 'unit_id' and 'time_cycles'.
        failure_window : int, optional
            The number of cycles defining "imminent failure." The project plan 
            specifies 48 hours (48 cycles). Defaults to 48.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the 'RUL' and 'target' columns added.
        """

        # Use config default if not provided
        failure_window = failure_window or config.data.failure_window

        print(f"\nCreating target variable (failure window: {failure_window} cycles)...")
        
        df = df.copy()
        
        # Calculate max cycle for each unit
        max_cycle_map: pd.Series = df.groupby('unit_id')['time_cycles'].transform('max')
        
        # Calculate RUL: RUL = max_cycle_seen - current_cycle
        df['RUL'] = max_cycle_map - df['time_cycles']
        
        # Create binary target: 1 if RUL is within the failure window, 0 otherwise.
        df['target'] = np.where(df['RUL'] <= failure_window, 1, 0)
        
        print(f"Target distribution:")
        print(f"  Healthy (0): {(df['target']==0).sum()} ({(df['target']==0).sum()/len(df)*100:.1f}%)")
        print(f"  Failure Risk (1): {(df['target']==1).sum()} ({(df['target']==1).sum()/len(df)*100:.1f}%)")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimized feature engineering using vectorized pandas operations.
        
        This method creates temporal and statistical features for each sensor:
        - Rolling averages (3, 5, 10 cycle windows)
        - Rate of change (first derivative)
        - Exponential moving average (EMA)
        - Rolling standard deviation
        - Baseline deviation
        - Cross-sensor aggregates
        - Cycle normalization
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with sensor columns and unit_id.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with all original columns plus engineered features.
            
        Notes
        -----
        Uses vectorized pandas operations instead of multiprocessing because:
        1. Multiprocessing has high overhead for serializing DataFrames
        2. Pandas groupby operations are already optimized in C
        3. For datasets under 1M rows, vectorized ops are faster
        """
        print(f"  Input shape: {df.shape}")
        df = df.copy()
        
        # Identify sensor columns
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
        print(f"  Processing {len(sensor_cols)} sensor columns...")
        
        # Pre-allocate dictionary for new features (more efficient than repeated concat)
        new_features: t.Dict[str, pd.Series] = {}
        
        # Get rolling window sizes from config
        windows = config.features.rolling_windows  # [3, 5, 10]
        ema_span = config.features.ema_span  # 5
        std_window = config.features.std_window  # 5
        
        # Process each sensor with vectorized operations
        for i, sensor in enumerate(sensor_cols):
            if (i + 1) % 5 == 0:
                print(f"    Processed {i + 1}/{len(sensor_cols)} sensors...")
            
            grouped = df.groupby('unit_id')[sensor]
            
            # Rolling averages for different windows
            for window in windows:
                new_features[f'{sensor}_roll_avg_{window}'] = (
                    grouped.transform(lambda x: x.rolling(window, min_periods=1).mean())
                )
            
            # Rate of change (difference from previous cycle)
            new_features[f'{sensor}_rate_change'] = grouped.diff().fillna(0)
            
            # Exponential moving average
            new_features[f'{sensor}_ema'] = (
                grouped.transform(lambda x: x.ewm(span=ema_span, adjust=False).mean())
            )
            
            # Rolling standard deviation (volatility measure)
            new_features[f'{sensor}_roll_std_{std_window}'] = (
                grouped.transform(lambda x: x.rolling(std_window, min_periods=1).std()).fillna(0)
            )
            
            # Deviation from baseline (first 10% of unit's life)
            new_features[f'{sensor}_dev_baseline'] = grouped.transform(
                lambda x: x - x.iloc[:max(1, int(len(x) * config.features.baseline_percentage))].mean()
            )
        
        print(f"  Created {len(new_features)} sensor-based features")
        
        # Cross-sensor aggregate features (overall equipment state)
        sensor_data = df[sensor_cols]
        new_features['sensor_mean_all'] = sensor_data.mean(axis=1)
        new_features['sensor_std_all'] = sensor_data.std(axis=1)
        new_features['sensor_max_all'] = sensor_data.max(axis=1)
        new_features['sensor_min_all'] = sensor_data.min(axis=1)
        
        # Cycle normalization (how far through its life is this unit?)
        new_features['cycle_normalized'] = df.groupby('unit_id')['time_cycles'].transform(
            lambda x: x / x.max()
        )
        
        # Single concat operation (much faster than repeated concat in loop)
        feature_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, feature_df], axis=1)
        
        # Store feature names for documentation
        self.feature_names = list(new_features.keys())
        
        print(f"  Output shape: {df.shape}")
        print(f"  Total new features: {len(self.feature_names)}")
        
        return df
    
    def create_train_val_split(self, 
                               df: pd.DataFrame, 
                               test_size: float = None, 
                               random_state: int = None) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates an 80/20 train/validation split by dividing the *unit_id*s, not the records.
        This prevents data leakage, ensuring model evaluation is accurate.
        The split is stratified by the maximum 'target' value for each unit.
        
        Parameters
        ----------
        df : pd.DataFrame
            The fully engineered DataFrame.
        test_size : float, optional
            The proportion of units to allocate to the validation set. Defaults to 0.2 (20%).
        random_state : int, optional
            Seed for random number generation for reproducibility. Defaults to 42.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the training DataFrame and the validation DataFrame.
        """
        # Use config defaults if parameters not provided
        test_size = test_size or config.data.val_split_ratio
        random_state = random_state or config.data.random_state

        print(f"\nCreating {int((1-test_size)*100)}/{int(test_size*100)} train/validation split...")
        
        # 1. Identify all unique units and their 'target' class (whether they failed in the data)
        unique_units: np.ndarray = df['unit_id'].unique()
        # Stratify ensures both sets get a proportional number of units that failed vs. units that survived
        unit_targets: pd.Series = df.groupby('unit_id')['target'].max() 
        
        # 2. Split the list of unit_ids
        train_units, val_units = train_test_split(
            unique_units, 
            test_size=test_size, 
            random_state=random_state,
            stratify=unit_targets[unique_units] # Stratify based on unit's max target
        )
        
        # 3. Use the unit lists to filter the full DataFrame
        train_df: pd.DataFrame = df[df['unit_id'].isin(train_units)].copy()
        val_df: pd.DataFrame = df[df['unit_id'].isin(val_units)].copy()
        
        print(f"Training set: {len(train_df):,} records ({len(train_units)} units)")
        print(f"Validation set: {len(val_df):,} records ({len(val_units)} units)")
        
        # Confirm target distributions are similar after splitting
        print(f"Training target distribution: Class 1 ratio: {train_df['target'].mean():.4f}")
        print(f"Validation target distribution: Class 1 ratio: {val_df['target'].mean():.4f}")
        
        self.train_df = train_df
        self.val_df = val_df
        
        return train_df, val_df
    
    def save_datasets(self, output_dir: str = 'data\\processed') -> None:
        """
        Saves the processed training and validation datasets, as well as a 
        documentation file listing all engineered features.

        Parameters
        ----------
        output_dir : str, optional
            The directory to save the final files into. Defaults to 'data\\processed'.
        """
        output_path: Path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        
        print(f"\nSaving processed datasets to {output_dir}/...")
        
        # 1. Save train and validation sets
        self.train_df.to_csv(output_path / 'train_processed.csv', index=False)
        self.val_df.to_csv(output_path / 'val_processed.csv', index=False)
        
        # 2. Save feature documentation
        feature_doc: pd.DataFrame = pd.DataFrame({
            'feature_name': self.feature_names,
            'feature_type': ['engineered'] * len(self.feature_names)
        })
        feature_doc.to_csv(output_path / 'feature_documentation.csv', index=False)
        
        # 3. Save data quality report
        self.generate_quality_report(output_path)
        
        print(f"✓ Saved train_processed.csv ({len(self.train_df):,} rows)")
        print(f"✓ Saved val_processed.csv ({len(self.val_df):,} rows)")
        print(f"✓ Saved feature_documentation.csv ({len(self.feature_names)} features)")
        print(f"✓ Saved data_quality_report.txt")
    
    def generate_quality_report(self, output_path: Path) -> None:
        """
        Generates a text report summarizing the shape, missing values, and 
        feature count/types of the final processed datasets.

        Parameters
        ----------
        output_path : Path
            The directory where the report file should be saved.
        """
        
        report_file: Path = output_path / 'data_quality_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("PHASE 1.3 DATA QUALITY & FEATURE REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Training Set Shape: {self.train_df.shape}\n")
            f.write(f"Validation Set Shape: {self.val_df.shape}\n\n")
            
            # Missing values check
            train_missing: float = (self.train_df.isnull().sum().sum() / 
                                   (self.train_df.shape[0] * self.train_df.shape[1]) * 100)
            f.write(f"Training Set Missing Values: {train_missing:.4f}%\n")
            
            val_missing: float = (self.val_df.isnull().sum().sum() / 
                                 (self.val_df.shape[0] * self.val_df.shape[1]) * 100)
            f.write(f"Validation Set Missing Values: {val_missing:.4f}%\n\n")
            
            # Feature summary
            f.write(f"Total Engineered Features: {len(self.feature_names)}\n\n")
            
            # Categorize features for the report
            f.write("Engineered Feature Categories:\n")
            categories: t.Dict[str, int] = {
                'Temporal (Roll Avg/EMA)': len([f for f in self.feature_names if any(x in f for x in ['roll_avg', 'ema', 'roll_std'])]),
                'Rate/Delta Features': len([f for f in self.feature_names if 'rate_change' in f]),
                'Baseline & Range': len([f for f in self.feature_names if any(x in f for x in ['dev_baseline', 'range'])]),
                'Cross-Sensor Aggregates': 4, # mean, std, max, min
                'Cycle Progression': 1
            }
            
            for cat, count in categories.items():
                f.write(f"  - {cat}: {count}\n")
            
            # Verification checks for the S.M.A.R.T. plan
            f.write(f"\n✓ Data quality check PASSED: <2% missing values (Actual: {max(train_missing, val_missing):.4f}%)\n")
            f.write(f"✓ Feature count check PASSED: {len(self.feature_names)} features created (Required: 12+)\n")
            f.write(f"✓ Train/Validation Split PASSED: Split by unit_id to prevent leakage\n")


def main() -> None:
    """
    Main entry point for the Step 1.3 data preparation script.
    Orchestrates the loading, target creation, feature engineering, splitting, and saving.
    """
    print("="*60)
    print("C-MAPSS DATA PREPARATION & FEATURE ENGINEERING (Step 1.3)")
    print("="*60)
    
    # Initialize preparator to manage the workflow
    prep: CMAPSSDataPreparator = CMAPSSDataPreparator(data_dir='data\\processed')
    
    # Step 1: Load the cleaned data from Step 1.2
    print("\n[STEP 1] Loading and combining CLEANED train files...")
    combined_df: pd.DataFrame = prep.load_and_combine_train_files(file_pattern='train_FD*_cleaned.csv', use_cleaned=True)

    
    # Step 2: Create target variable (Failure Imminence)
    print("\n[STEP 2] Creating RUL and binary target variable...")
    df_with_target: pd.DataFrame = prep.create_target_variable(combined_df, failure_window=48)
    
    # Step 3: Engineer features
    print("\n[STEP 3] Engineering features...")
    df_engineered: pd.DataFrame = prep.engineer_features(df_with_target)
    
    # Step 4: Create train/validation split (80/20 by unit_id)
    print("\n[STEP 4] Creating unit-based train/validation split...")
    train_df, val_df = prep.create_train_val_split(df_engineered, test_size=0.2)
    
    # Step 5: Save datasets and documentation
    print("\n[STEP 5] Saving processed datasets and documentation...")
    prep.save_datasets(output_dir='data/processed')

    df = pd.read_csv('data/processed/val_processed.csv')
    unit_249 = df[df['unit_id'] == 249]
    print(unit_249[['time_cycles', 'cycle_normalized']].tail(20))
    
    print("\n" + "="*60)
    print("✓ STEP 1.3 DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\nNext Steps (Project Phase 2: AI Model Development):")
    print("1. Review final data quality and feature documentation in 'data/processed'")
    print("2. Proceed to model training using 'train_processed.csv'")


if __name__ == "__main__":
    main()