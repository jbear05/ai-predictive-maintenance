"""
Centralized configuration management for predictive maintenance project.

This module provides a single source of truth for all configuration parameters,
preventing hardcoded values scattered across multiple files.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import json


@dataclass
class PathConfig:
    """File system paths configuration."""
    
    # Root directories
    project_root: Path = Path(__file__).parent.parent.parent
    data_root: Path = field(default_factory=lambda: Path("data"))
    models_root: Path = field(default_factory=lambda: Path("models"))
    results_root: Path = field(default_factory=lambda: Path("results"))
    
    @property
    def raw_data(self) -> Path:
        return self.data_root / "raw"
    
    @property
    def processed_data(self) -> Path:
        return self.data_root / "processed"
    
    @property
    def train_file(self) -> Path:
        return self.processed_data / "train_processed.csv"
    
    @property
    def val_file(self) -> Path:
        return self.processed_data / "val_processed.csv"
    
    def ensure_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        for path in [self.data_root, self.models_root, self.results_root,
                     self.raw_data, self.processed_data]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data processing configuration."""
    
    # Column definitions
    column_names: List[str] = field(default_factory=lambda: [
        'unit_id', 'time_cycles', 
        'setting_1', 'setting_2', 'setting_3',
    ] + [f'sensor_{i}' for i in range(1, 22)])
    
    # Columns to exclude from features
    non_feature_columns: List[str] = field(default_factory=lambda: [
        'target', 'unit_id', 'source_file', 'RUL', 'time_cycles'
    ])
    
    # Data quality thresholds
    missing_value_threshold: float = 0.02  # 2% max missing values
    outlier_sigma: float = 3.0  # 3-sigma rule
    min_variance: float = 1e-10  # Skip constant features
    
    # Target variable
    failure_window: int = 48  # Cycles before failure (RUL threshold)
    
    # Train/validation split
    val_split_ratio: float = 0.2
    random_state: int = 42


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    
    # Rolling window sizes
    rolling_windows: List[int] = field(default_factory=lambda: [3, 5, 10])
    
    # EMA span
    ema_span: int = 5
    
    # Standard deviation window
    std_window: int = 5
    
    # Baseline calculation (% of early cycles)
    baseline_percentage: float = 0.1


@dataclass
class ModelConfig:
    """Model training configuration."""
    
    # XGBoost default hyperparameters
    xgboost_params: dict = field(default_factory=lambda: {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'random_state': 42,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    })
    
    # Grid search parameter grid
    param_grid: dict = field(default_factory=lambda: {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3]
    })
    
    # Quick test grid (for development)
    quick_param_grid: dict = field(default_factory=lambda: {
        'n_estimators': [100],
        'max_depth': [3, 5],
        'learning_rate': [0.1]
    })
    
    # Grid search configuration
    cv_folds: int = 3
    scoring_metric: str = 'recall'  # Optimize for recall
    n_jobs: int = -1  # Use all CPU cores
    
    # Performance targets
    target_accuracy: float = 0.80
    target_recall: float = 0.85
    target_precision: float = 0.70


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Global settings
    verbose: bool = True
    use_quick_grid: bool = False  # Toggle for faster development
    
    def __post_init__(self):
        """Initialize directories after config creation."""
        self.paths.ensure_directories()
    
    @classmethod
    def from_json(cls, filepath: Path) -> 'Config':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, filepath: Path) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            'paths': self.paths.__dict__,
            'data': self.data.__dict__,
            'features': self.features.__dict__,
            'model': self.model.__dict__,
            'verbose': self.verbose,
            'use_quick_grid': self.use_quick_grid
        }
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def get_active_param_grid(self) -> dict:
        """Get the appropriate parameter grid based on mode."""
        if self.use_quick_grid:
            return self.model.quick_param_grid
        return self.model.param_grid


# Singleton instance - import this in other modules
config = Config()


# Example usage in your scripts:
if __name__ == "__main__":
    # Initialize config
    cfg = Config()
    
    # Access paths
    print(f"Training data: {cfg.paths.train_file}")
    print(f"Models directory: {cfg.paths.models_root}")
    
    # Access parameters
    print(f"Failure window: {cfg.data.failure_window} cycles")
    print(f"Target recall: {cfg.model.target_recall}")
    
    # Toggle development mode
    cfg.use_quick_grid = True
    print(f"Using grid: {cfg.get_active_param_grid()}")
    
    # Save config for reproducibility
    cfg.to_json(Path("config_snapshot.json"))