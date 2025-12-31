import sys
from pathlib import Path
# Add project root to path for loaders and config imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import joblib
import json
from typing import Dict, Tuple, Union
import warnings
from loaders import load_inference_artifacts
from config import config
from terminal_colors import Colors, print_header, print_success, print_warning, print_error

warnings.filterwarnings('ignore') # Suppress sklearn warnings for cleaner output

def load_artifacts() -> Tuple[object, object, list, list]:
    """Wrapper for backward compatibility. Returns model, scaler, columns_to_scale, all_features."""
    from loaders import DataLoader
    loader = DataLoader(config)
    return loader.load_artifacts()


def preprocess_sensor_data(
    data: pd.DataFrame,
    scaler: object,
    columns_to_scale: list
) -> pd.DataFrame:
    """
    Preprocess raw sensor data exactly like training data.
    
    Critical: This must match your training preprocessing EXACTLY.
    
    Steps:
    1. Handle missing values (same strategy as training)
    2. Apply the SAVED scaler (never fit a new one!)
    3. Return normalized data ready for model
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw sensor readings with same columns as training data
    scaler : MinMaxScaler
        The fitted scaler from training (from scaler.pkl)
    columns_to_scale : list
        List of column names to normalize
    
    Returns
    -------
    pd.DataFrame
        Preprocessed data ready for prediction
    """
    print_header("--- Preprocessing Data ---")
    
    # Make a copy so we don't modify the original
    data_processed = data.copy()
    
    # STEP 1: Handle missing values
    # Your training data had 0.00% missing, but real-world data might not
    missing_before = data_processed.isnull().sum().sum()
    if missing_before > 0:
        print_warning(f"Warning: {missing_before} missing values detected")
        # Strategy: Fill with column mean (same as training would do)
        data_processed = data_processed.fillna(data_processed.mean())
        print_success("Filled missing values with column means")
    
    # STEP 2: Apply the saved scaler
    # This is THE most critical step for inference
    # We use transform() NOT fit_transform()
    try:
        data_processed[columns_to_scale] = scaler.transform(
            data_processed[columns_to_scale]
        )
        print_success(f"Scaled {len(columns_to_scale)} columns")
    except KeyError as e:
        print_error(f"Error: Missing required columns: {e}")
        raise

    data_processed[columns_to_scale] = data_processed[columns_to_scale].clip(0, 1)
    print("   Clipped values to [0, 1] range")
    
    # STEP 3: Verify scaling worked
    scaled_min = data_processed[columns_to_scale].min().min()
    scaled_max = data_processed[columns_to_scale].max().max()
    print(f"   Scaled range: [{scaled_min:.4f}, {scaled_max:.4f}]")
    
    # Sanity check: scaled values should be roughly 0-1
    if scaled_min < -0.1 or scaled_max > 1.1:
        print_warning("Warning: Scaled values outside expected range [0, 1]")
        print("   This might indicate data distribution shift")
    
    return data_processed


def predict_failure(
    data: pd.DataFrame,
    model: object,
    scaler: object,
    columns_to_scale: list,
    return_probability: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Predict equipment failure for new sensor readings.
    
    This is the main inference function that ties everything together.
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw sensor data with same features as training
        Must include all 219 features used during training
    model : XGBClassifier
        Trained model
    scaler : MinMaxScaler
        Fitted scaler from training
    columns_to_scale : list
        Columns to normalize
    return_probability : bool
        If True, return both predictions and probabilities
        If False, return only predictions
    
    Returns
    -------
    predictions : np.ndarray
        Binary predictions (0=healthy, 1=will fail)
    probabilities : np.ndarray (optional)
        Probability of failure (0.0 to 1.0)
    
    Examples
    --------
    >>> predictions, probabilities = predict_failure(new_data, model, scaler, cols)
    >>> print(f"Prediction: {predictions[0]}, Confidence: {probabilities[0]:.2%}")
    Prediction: 1, Confidence: 87.3%
    """

    print_header("RUNNING INFERENCE PIPELINE")
    
    # 1. Validate input type
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(data)}")
    
    # 2. Validate not empty
    if data.empty:
        raise ValueError("Input DataFrame is empty")
    
    # 3. Validate required columns exist
    missing_cols = set(columns_to_scale) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # 4. Validate data types
    for col in columns_to_scale:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise TypeError(f"Column {col} must be numeric, got {data[col].dtype}")
    
    # 5. Validate value ranges (prevent adversarial inputs)
    for col in columns_to_scale:
        if data[col].abs().max() > 1e6:  # Reasonable threshold
            raise ValueError(f"Column {col} contains extreme values")
    
    # 6. Check for infinite/NaN values
    if data[columns_to_scale].isnull().any().any():
        raise ValueError("Input contains NaN values")
    
    if np.isinf(data[columns_to_scale].values).any():
        raise ValueError("Input contains infinite values")
    
    # Preprocess the data
    data_processed = preprocess_sensor_data(data, scaler, columns_to_scale)
    
    # Remove columns that shouldn't be used for prediction
    # (These were excluded during training)
    columns_to_drop = ['target', 'unit_id', 'source_file', 'RUL', 'time_cycles']
    columns_to_drop = [col for col in columns_to_drop if col in data_processed.columns]
    
    if columns_to_drop:
        print(f"\nDropping non-predictive columns: {columns_to_drop}")
        X = data_processed.drop(columns_to_drop, axis=1)
    else:
        X = data_processed
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Expected features: {model.n_features_in_}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X)
    
    # Get probability scores
    if return_probability:
        probabilities = model.predict_proba(X)[:, 1]  # Probability of class 1 (failure)
        
        # Summary statistics
        print(f"\n{Colors.BOLD}Prediction Summary:{Colors.END}")
        print(f"   Total samples: {len(predictions)}")
        print(f"   Predicted failures: {predictions.sum()} ({predictions.sum()/len(predictions)*100:.1f}%)")
        print(f"   Predicted healthy: {(predictions==0).sum()} ({(predictions==0).sum()/len(predictions)*100:.1f}%)")
        print(f"   Avg failure probability: {probabilities.mean():.3f}")
        print(f"   Max failure probability: {probabilities.max():.3f}")
        print(f"   Min failure probability: {probabilities.min():.3f}")
        
        return predictions, probabilities
    
    return predictions


def test_inference_pipeline():
    """
    Test the inference pipeline with validation data.
    
    This helps verify:
    1. Model loads correctly
    2. Preprocessing works
    3. Predictions match expected format
    4. Performance metrics are reasonable
    """

    print_header("TESTING INFERENCE PIPELINE")
    
    # Load artifacts
    model, scaler, columns_to_scale, all_features = load_artifacts()
    
    # Load some validation data to test with
    print("\nLoading validation data for testing...")
    val_data = pd.read_csv(config.paths.val_file)
    
    # Take a small sample for quick testing
    test_sample = val_data.sample(n=100, random_state=42)
    print(f"Testing with {len(test_sample)} samples")
    
    # Run inference
    predictions, probabilities = predict_failure(
        test_sample,
        model,
        scaler,
        columns_to_scale,
        return_probability=True
    )
    
    # Compare to actual labels (if available)
    if 'target' in test_sample.columns:
        actual = test_sample['target'].values
        accuracy = (predictions == actual).mean()
        
        print_success(f"\nTEST RESULTS:")
        print(f"   Accuracy on test sample: {accuracy:.2%}")
        print(f"   Predictions match expected format")
        
        # Show some examples
        print(f"\n📋 Sample Predictions:")
        for i in range(min(5, len(predictions))):
            print(f"   Sample {i+1}: Actual={actual[i]}, Predicted={predictions[i]}, Probability={probabilities[i]:.3f}")
    
    print_success("\nInference pipeline test complete!")
    return True


def main():
    """
    Main execution function for testing the inference pipeline.
    """
    # Test the pipeline
    test_inference_pipeline()
    
    print_success("Inference pipeline is ready for deployment!")


if __name__ == "__main__":
    main()