"""
Centralized data loading utilities.

This module eliminates repeated data loading logic across scripts
and provides a consistent interface for accessing training data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import logging


class DataLoader:
    """
    Handles all data loading operations for the project.
    
    Benefits:
    - Single source of truth for data loading
    - Consistent error handling
    - Easy to mock for testing
    - Caching support for faster development
    """
    
    def __init__(self, config=None, use_cache: bool = True):
        """
        Initialize data loader.
        
        Parameters
        ----------
        config : Config, optional
            Configuration object. If None, uses default config.
        use_cache : bool
            Whether to cache loaded DataFrames in memory.
        """
        self.config = config
        self.use_cache = use_cache
        self._cache = {}
        self.logger = logging.getLogger(__name__)
    
    def load_raw_cmapss_files(
        self, 
        file_pattern: str = "train_FD*.txt"
    ) -> pd.DataFrame:
        """
        Load and combine raw C-MAPSS dataset files.
        
        Parameters
        ----------
        file_pattern : str
            Glob pattern to match files (e.g., 'train_FD*.txt')
        
        Returns
        -------
        pd.DataFrame
            Combined dataframe with 'source_file' column added
        
        Raises
        ------
        FileNotFoundError
            If no files matching pattern are found
        """
        cache_key = f"raw_{file_pattern}"
        if self.use_cache and cache_key in self._cache:
            self.logger.info(f"Loading {cache_key} from cache")
            return self._cache[cache_key].copy()
        
        # Get data directory from config
        data_dir = self.config.paths.raw_data if self.config else Path("data/raw")
        
        # Find matching files
        files = list(data_dir.glob(file_pattern))
        
        if not files:
            raise FileNotFoundError(
                f"No files found matching '{file_pattern}' in {data_dir}"
            )
        
        self.logger.info(f"Found {len(files)} files to load")
        
        # Load and combine
        all_dfs = []
        for file in sorted(files):
            self.logger.info(f"Loading {file.name}...")
            
            # Get column names from config
            columns = (self.config.data.column_names 
                      if self.config 
                      else self._default_column_names())
            
            df = pd.read_csv(file, sep=r'\s+', header=None, names=columns)
            df['source_file'] = file.stem  # Track origin
            all_dfs.append(df)
        
        combined = pd.concat(all_dfs, ignore_index=True)
        
        if self.use_cache:
            self._cache[cache_key] = combined.copy()
        
        self.logger.info(f"Loaded {len(combined):,} total records")
        return combined
    
    def load_processed_split(
        self,
        split: str = "train"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load processed training or validation data with features separated from target.
        
        Parameters
        ----------
        split : str
            Either 'train' or 'val'
        
        Returns
        -------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        
        Raises
        ------
        ValueError
            If split is not 'train' or 'val'
        FileNotFoundError
            If processed file doesn't exist
        """
        if split not in ['train', 'val']:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")
        
        cache_key = f"processed_{split}"
        if self.use_cache and cache_key in self._cache:
            self.logger.info(f"Loading {cache_key} from cache")
            X, y = self._cache[cache_key]
            return X.copy(), y.copy()
        
        # Get file path from config
        if self.config:
            file_path = (self.config.paths.train_file 
                        if split == 'train' 
                        else self.config.paths.val_file)
        else:
            file_path = Path(f"data/processed/{split}_processed.csv")
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Processed {split} file not found: {file_path}\n"
                f"Run data preparation pipeline first."
            )
        
        self.logger.info(f"Loading {split} data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Separate features from target
        X, y = self._prepare_features_target(df)
        
        if self.use_cache:
            self._cache[cache_key] = (X.copy(), y.copy())
        
        self.logger.info(f"Loaded {split}: {X.shape[0]:,} samples, {X.shape[1]} features")
        return X, y
    
    def _prepare_features_target(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features from target and drop non-predictive columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Full dataframe with all columns
        
        Returns
        -------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        """
        # Get columns to drop from config
        cols_to_drop = (self.config.data.non_feature_columns 
                       if self.config 
                       else ['target', 'unit_id', 'source_file', 'RUL', 'time_cycles'])
        
        # Only drop columns that exist
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        
        # Extract target
        if 'target' not in df.columns:
            raise ValueError("DataFrame must contain 'target' column")
        
        y = df['target'].copy()
        
        # Create feature matrix
        X = df.drop(cols_to_drop, axis=1)
        
        return X, y
    
    def load_artifacts(self) -> Tuple[object, object, List[str]]:
        """
        Load trained model artifacts (model, scaler, columns).
        
        Returns
        -------
        model : object
            Trained model (e.g., XGBClassifier)
        scaler : object
            Fitted scaler (e.g., MinMaxScaler)
        columns_to_scale : List[str]
            List of column names that should be scaled
        
        Raises
        ------
        FileNotFoundError
            If any artifact files are missing
        """
        import joblib
        import json
        
        models_dir = self.config.paths.models_root if self.config else Path("models")
        
        # Check all files exist
        required_files = {
            'model': models_dir / 'xgboost_model.pkl',
            'scaler': models_dir / 'scaler.pkl',
            'columns': models_dir / 'scaler_columns.json'
        }
        
        for name, path in required_files.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {name} artifact: {path}")
        
        self.logger.info("Loading model artifacts...")
        
        # Load model
        model = joblib.load(required_files['model'])
        self.logger.info(f"✅ Model loaded (expects {model.n_features_in_} features)")
        
        # Load scaler
        scaler = joblib.load(required_files['scaler'])
        self.logger.info("✅ Scaler loaded")
        
        # Load column list
        with open(required_files['columns'], 'r') as f:
            columns_to_scale = json.load(f)
        self.logger.info(f"✅ Scaler columns loaded ({len(columns_to_scale)} columns)")
        
        return model, scaler, columns_to_scale
    
    def clear_cache(self) -> None:
        """Clear the in-memory data cache."""
        self._cache.clear()
        self.logger.info("Cache cleared")
    
    @staticmethod
    def _default_column_names() -> List[str]:
        """Fallback column names if config not available."""
        return ['unit_id', 'time_cycles', 
                'setting_1', 'setting_2', 'setting_3'] + \
               [f'sensor_{i}' for i in range(1, 22)]


# Convenience functions for backward compatibility
def load_train_data(config=None) -> Tuple[pd.DataFrame, pd.Series]:
    """Quick function to load training data."""
    loader = DataLoader(config)
    return loader.load_processed_split('train')


def load_val_data(config=None) -> Tuple[pd.DataFrame, pd.Series]:
    """Quick function to load validation data."""
    loader = DataLoader(config)
    return loader.load_processed_split('val')


def load_inference_artifacts(config=None) -> Tuple[object, object, List[str]]:
    """Quick function to load model artifacts."""
    loader = DataLoader(config)
    return loader.load_artifacts()


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize loader
    loader = DataLoader()
    
    # Load training data
    X_train, y_train = loader.load_processed_split('train')
    print(f"Training data: {X_train.shape}")
    
    # Load validation data
    X_val, y_val = loader.load_processed_split('val')
    print(f"Validation data: {X_val.shape}")
    
    # Load model artifacts
    model, scaler, cols = loader.load_artifacts()
    print(f"Model ready for inference")