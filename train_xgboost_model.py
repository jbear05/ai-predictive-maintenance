from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import joblib


def train_xgboost_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> XGBClassifier:
    """
    Train and evaluate an XGBoost classifier with hyperparameter tuning.
    
    This function performs comprehensive hyperparameter optimization using GridSearchCV
    to find the best XGBoost model configuration for predictive maintenance. The model
    is optimized for recall to minimize false negatives (missed failures), which is
    critical in maintenance applications.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix containing sensor readings and engineered features.
        Shape: (n_samples, n_features)
    y_train : pd.Series
        Training target vector with binary labels:
        - 0: Equipment healthy (will not fail within prediction window)
        - 1: Equipment will fail (within next 48 operational cycles)
        Shape: (n_samples,)
    X_test : pd.DataFrame
        Test feature matrix for model evaluation.
        Must have same features as X_train in same order.
        Shape: (m_samples, n_features)
    y_test : pd.Series
        Test target vector with binary labels (0: healthy, 1: failure).
        Shape: (m_samples,)
    
    Returns
    -------
    XGBClassifier
        The best performing XGBoost model found during grid search, already fitted
        on the training data.
    
    Notes
    -----
    **Class Imbalance Handling:**
    The function automatically calculates `scale_pos_weight` to handle imbalanced
    datasets where failures (class 1) are less frequent than healthy samples (class 0).
    
    **Hyperparameter Grid:**
    The grid search explores 108 combinations (3×3×3×2×2×2) of:
    - n_estimators: [100, 200, 300] - Number of boosting rounds
    - max_depth: [3, 5, 7] - Maximum tree depth
    - learning_rate: [0.01, 0.1, 0.2] - Boosting learning rate
    - subsample: [0.8, 1.0] - Fraction of samples used per tree
    - colsample_bytree: [0.8, 1.0] - Fraction of features used per tree
    - min_child_weight: [1, 3] - Minimum sum of instance weights in child node
    
    **Optimization Metric:**
    Models are optimized for recall (sensitivity) to prioritize catching failures
    over minimizing false alarms. This aligns with predictive maintenance goals
    where missing a failure is costlier than a false alarm.
    
    **Performance Targets:**
    - Accuracy: ≥80%
    - Recall: ≥85% (prioritized)
    - Precision: ≥70%
    
    **Output Files:**
    - models/xgboost_model.pkl: Serialized trained model
    - results/model_comparison.txt: Performance metrics (appended)
    
    **Computational Notes:**
    - Uses 3-fold cross-validation for each hyperparameter combination
    - Total model trainings: 108 combinations × 3 folds = 324 fits
    - Estimated runtime: 1-3 hours depending on dataset size and CPU cores
    - Uses all available CPU cores (n_jobs=-1) for parallel processing
    
    """
    
    # See how imbalanced your data is
    unique : np.ndarray
    counts : np.ndarray
    
    unique , counts = np.unique(y_train, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")

    # Calculate scale_pos_weight for XGBoost
    negative_cases : int = counts[0]
    positive_cases : int = counts[1]
    scale_pos_weight : float = negative_cases / positive_cases
    print(f"Scale pos weight: {scale_pos_weight}")

    # Create parameter combinations to test (20+ combinations)
    param_grid : dict = {
        'n_estimators': [100, 200, 300],           # Number of trees
        'max_depth': [3, 5, 7],                    # Tree depth
        'learning_rate': [0.01, 0.1, 0.2],         # Step size
        'subsample': [0.8, 1.0],                   # Sample % per tree
        'colsample_bytree': [0.8, 1.0],            # Feature % per tree
        'min_child_weight': [1, 3]                 # Min samples in leaf
    }

    # param_grid = {
    #     'n_estimators': [100, 200],
    #     'max_depth': [3, 5, 7],
    #     'learning_rate': [0.1, 0.2],
    #     'subsample': [0.8],
    #     'colsample_bytree': [0.8],
    # }
    # # ^This gives 2*3*2 = 12 combinations for faster testing

    # # Quick test grid (will take ~3-6 minutes)
    # param_grid = {
    #     'n_estimators': [100],
    #     'max_depth': [3, 5],
    #     'learning_rate': [0.1],
    # }
    # # This is only 2 combinations × 3 folds = 6 trainings
    # # Use for finding out how long grid search takes


    # Create base model with fixed parameters
    xgb_base = XGBClassifier(
        scale_pos_weight=scale_pos_weight,  # Handle imbalance
        random_state=42,                     # Reproducibility
        eval_metric='logloss',               # Evaluation metric
        use_label_encoder=False              # Avoid warning
    )


    # Configure grid search
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='recall',        # Optimize for recall (≥85% target)
        cv=3,                    # 3-fold cross-validation
        verbose=2,               # Show progress
        n_jobs=-1                # Use all CPU cores
    )


    print("Starting grid search... This may take 30+ minutes")
    grid_search.fit(X_train, y_train)
    print("Grid search complete!")


    # Extract best configuration
    best_params : dict = grid_search.best_params_
    print(f"\nBest parameters found:")
    print(best_params)

    # Get the best model
    best_model : XGBClassifier = grid_search.best_estimator_


    # Make predictions
    y_pred : np.ndarray = best_model.predict(X_test)

    # Calculate metrics
    accuracy : float = accuracy_score(y_test, y_pred)
    recall : float = recall_score(y_test, y_pred)
    precision : float = precision_score(y_test, y_pred)

    # Save trained model as pickle file for later use
    joblib.dump(best_model, 'models/xgboost_model.pkl')

    print(f"\n=== MODEL PERFORMANCE ===")
    print(f"Accuracy:  {accuracy:.2%} (Target: ≥80%)")
    print(f"Recall:    {recall:.2%} (Target: ≥85%)")
    print(f"Precision: {precision:.2%} (Target: ≥70%)")

    # Detailed classification report
    print("\n" + classification_report(y_test, y_pred))

    # Save results to file
    with open('results/model_comparison.txt', 'a') as f:
        f.write("\n=== XGBOOST MODEL ===\n")
        f.write(f"Best Parameters: {best_params}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Target Met: {'YES' if accuracy >= 0.80 and recall >= 0.85 and precision >= 0.70 else 'NO'}\n")

    return best_model


def main():
    """
    Execute the complete XGBoost model training pipeline.
    
    This function orchestrates the full workflow for training an XGBoost-based
    predictive maintenance model, including data loading, preprocessing validation,
    and model training with hyperparameter optimization.
    
    Workflow
    --------
    1. Load pre-split training and validation datasets from CSV files
    2. Remove non-predictive columns and potential data leakage sources
    3. Separate features (X) from target variable (y)
    4. Display dataset statistics for verification
    5. Train XGBoost model with grid search hyperparameter tuning
    6. Evaluate model performance on holdout test set
    7. Save trained model and performance metrics
    
    Expected Directory Structure
    ----------------------------
    Project Root/
    ├── data/
    │   └── processed/
    │       ├── train_processed.csv    # Training dataset (required)
    │       └── val_processed.csv      # Validation dataset (required)
    ├── models/                        # Model artifacts (created if missing)
    ├── results/                       # Performance reports (created if missing)
    └── train_xgboost.py              # This script
    
    Input Files
    -----------
    data/processed/train_processed.csv : CSV file
        Processed training dataset containing:
        - Sensor readings (sensor_1 through sensor_21)
        - Operational settings (setting_1, setting_2, setting_3)
        - Engineered features (rolling averages, rate of change, etc.)
        - Target variable ('target': 0=healthy, 1=will fail)
        - Metadata columns (unit_id, RUL, time_cycles, source_file)
    
    data/processed/val_processed.csv : CSV file
        Processed validation dataset with identical structure to training data.
        Used for unbiased model evaluation.
    
    Output Files
    ------------
    models/xgboost_model.pkl : Pickle file
        Serialized trained XGBoost model, ready for deployment or inference.
        Can be loaded with: `model = joblib.load('models/xgboost_model.pkl')`
    
    results/model_comparison.txt : Text file
        Performance metrics report (appended, not overwritten) containing:
        - Best hyperparameters found
        - Accuracy, recall, and precision scores
        - Whether performance targets were met

    Notes
    -----
    Columns dropped:
    - 'target': The prediction target (cannot be used as a feature)
    - 'unit_id': Equipment identifier (non-predictive)
    - 'source_file': Data source metadata (non-predictive)
    - 'RUL': Remaining Useful Life (would leak future information)
    - 'time_cycles': Temporal index (non-predictive, sequential identifier)

    **Prediction Window:**
    The target variable indicates whether equipment will fail within the next
    48 operational cycles (approximately 1-2 weeks). Each cycle represents
    one complete flight operation for turbofan engines.
    
    **Class Imbalance:**
    Failure events are minority class (~10% of samples). The training function
    automatically handles this using XGBoost's scale_pos_weight parameter.
    
    **Computational Requirements:**
    - RAM: ~4-8 GB for datasets with 100K+ samples
    - CPU: Multi-core processor recommended (uses all available cores)
    - Time: 1-3 hours for full grid search with 108 parameter combinations
    
    **Performance Targets:**
    - Accuracy ≥ 80%
    - Recall ≥ 85% (prioritized to minimize missed failures)
    - Precision ≥ 70%
    
    Raises
    ------
    FileNotFoundError
        If required input CSV files are not found in data/processed/ directory.
    
    KeyError
        If expected columns are missing from input datasets.
    
    ValueError
        If datasets contain incompatible dtypes or shapes.
    """
    # Load pre-split training and validation datasets
    train_df: pd.DataFrame = pd.read_csv('data/processed/train_processed.csv')
    val_df: pd.DataFrame = pd.read_csv('data/processed/val_processed.csv')
    
    # Define columns to exclude from features
    # These are either non-predictive, identifiers, or leak future information
    columns_to_drop: list = ['target', 'unit_id', 'source_file', 'RUL', 'time_cycles']
    
    # Prepare feature matrices (X) and target vectors (y)
    X_train: pd.DataFrame = train_df.drop(columns_to_drop, axis=1)
    y_train: pd.Series = train_df['target']
    
    X_test: pd.DataFrame = val_df.drop(columns_to_drop, axis=1)
    y_test: pd.Series = val_df['target']
    
    # Display dataset information
    print(f"✅ Train: {len(X_train)} samples")
    print(f"✅ Val: {len(X_test)} samples")
    print(f"✅ Features: {X_train.shape[1]} columns")
    
    # Train and evaluate XGBoost model
    train_xgboost_model(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()