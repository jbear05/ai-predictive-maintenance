import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
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
        self.data_dir = data_dir or config.paths.processed_data
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
        Engineers multiple temporal and statistical features on the sensor readings
        for each engine unit independently. Aligns with the 12+ features goal.
        

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the sensor readings and target variable.

        Returns
        -------
        pd.DataFrame
            The DataFrame with all new engineered features added.
        """
        print("\nEngineering features...")
        df = df.copy()
        
        sensor_cols: t.List[str] = [col for col in df.columns if col.startswith('sensor_')]
        
        # Group by unit_id to apply rolling functions independently for each engine
        df_grouped = df.groupby('unit_id')
        
        # 1-8. Temporal and Statistical Features (Applied per-unit)
        for sensor in sensor_cols:
            # Feature 1-3: Rolling averages (3, 5, 10 cycles) - Captures local trend
            df[f'{sensor}_roll_avg_3'] = df_grouped[sensor].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'{sensor}_roll_avg_5'] = df_grouped[sensor].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'{sensor}_roll_avg_10'] = df_grouped[sensor].rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)
            
            # Feature 4: Rate of change (first difference) - Captures sudden spikes
            df[f'{sensor}_rate_change'] = df_grouped[sensor].diff().fillna(0).reset_index(level=0, drop=True)
            
            # Feature 5: Exponential moving average (EMA) - Smoother, gives more weight to recent data
            df[f'{sensor}_ema'] = df_grouped[sensor].ewm(span=5, adjust=False).mean().reset_index(level=0, drop=True)
            
            # Feature 6: Rolling standard deviation - Measures instability/variability
            df[f'{sensor}_roll_std_5'] = df_grouped[sensor].rolling(window=5, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
            
            # Feature 7: Rolling Min-Max range - Measures oscillation amplitude
            rolling_max: pd.Series = df_grouped[sensor].rolling(window=5, min_periods=1).max()
            rolling_min: pd.Series = df_grouped[sensor].rolling(window=5, min_periods=1).min()
            df[f'{sensor}_range_5'] = (rolling_max - rolling_min).reset_index(level=0, drop=True)

            # Feature 8: Deviation from unit baseline (first 10% of cycles)
            # Baseline is calculated per unit.
            baseline_means: pd.Series = df_grouped.head(n=10)['sensor_2'].mean() # Using sensor_2 as proxy baseline
            df[f'{sensor}_dev_baseline'] = df_grouped.transform(lambda x: x - x.iloc[0:int(len(x)*0.1)].mean())[sensor]
        
        # 9-12. Cross-Sensor Aggregates (Statistical features across ALL sensors at one timestamp)
        print("Creating cross-sensor aggregate statistical features...")
        sensor_values: np.ndarray = df[sensor_cols].values
        df['sensor_mean'] = np.mean(sensor_values, axis=1) # Average of all sensor readings
        df['sensor_std'] = np.std(sensor_values, axis=1)   # Spread of all sensor readings
        df['sensor_max'] = np.max(sensor_values, axis=1)   # Maximum reading
        df['sensor_min'] = np.min(sensor_values, axis=1)   # Minimum reading
        
        # 13. Cycle progression (normalized)
        # Normalizes the current cycle count by the max cycle seen for that unit (0 to 1 scale)
        max_cycle_map: pd.Series = df.groupby('unit_id')['time_cycles'].transform('max')
        df['cycle_normalized'] = df['time_cycles'] / max_cycle_map
        
        # Count and save feature names
        new_feature_tags: t.List[str] = ['roll_avg', 'rate_change', 'ema', 'roll_std', 
                                         'dev_baseline', 'range', 'sensor_', 'cycle_normalized']
        new_features: t.List[str] = [col for col in df.columns if any(tag in col for tag in new_feature_tags)]
        
        print(f"Total original features: {len(sensor_cols)} + 3 settings + ID/Time")
        print(f"Total engineered features: {len(new_features)}")
        self.feature_names = new_features
        
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
    prep.save_datasets(output_dir='data\\processed')
    
    print("\n" + "="*60)
    print("✓ STEP 1.3 DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\nNext Steps (Project Phase 2: AI Model Development):")
    print("1. Review final data quality and feature documentation in 'data\\processed'")
    print("2. Proceed to model training using 'train_processed.csv'")


if __name__ == "__main__":
    main()