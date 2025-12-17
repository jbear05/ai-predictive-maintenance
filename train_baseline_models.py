import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import time


def train_baseline_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Train and evaluate baseline machine learning models for predictive maintenance.
    
    This function trains two baseline models (Logistic Regression and Random Forest),
    evaluates their performance on test data, saves the trained models, and generates
    a comparison report.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix containing sensor readings and engineered features.
        Shape: (n_train_samples, n_features)
    y_train : pd.Series
        Training target variable (binary: 0 = no failure, 1 = will fail).
        Shape: (n_train_samples,)
    X_test : pd.DataFrame
        Test feature matrix for model evaluation.
        Shape: (n_test_samples, n_features)
    y_test : pd.Series
        Test target variable for model evaluation.
        Shape: (n_test_samples,)
    
    Returns
    -------
    None
        This function saves models to disk and writes results to a text file.
        Models are saved to: 'models/logistic_model.pkl' and 'models/random_forest_model.pkl'
        Results are saved to: 'results/model_comparison.txt'
    
    Notes
    -----
    - Both models use class_weight='balanced' to handle class imbalance
    - Logistic Regression uses SAGA solver with 2000 max iterations
    - Random Forest uses 100 estimators with all CPU cores (n_jobs=-1)
    - Performance metrics include: accuracy, precision, recall, and F1-score
    
    """
    print("🚀 Starting baseline model training...\n")

    # ========== Logistic Regression Training ==========
    print("\n🚀 Training Logistic Regression...")
    start_time = time.time()

    # Create the logistic regression model
    # - class_weight='balanced': Adjusts weights to handle class imbalance
    # - max_iter=2000: Maximum number of iterations for convergence
    # - solver='saga': Stochastic Average Gradient descent solver (handles large datasets well)
    log_reg : LogisticRegression = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=2000,
        solver='saga'
    )

    # Train the model on training data
    log_reg.fit(X_train, y_train)

    # Record training time
    log_reg_time : float = time.time() - start_time
    print(f"✅ Training completed in {log_reg_time:.2f} seconds")

    # Make predictions on test set
    y_pred_log = log_reg.predict(X_test)

    # Calculate performance metrics
    log_accuracy = accuracy_score(y_test, y_pred_log)      # Overall correctness
    log_precision = precision_score(y_test, y_pred_log)    # True positives / All positive predictions
    log_recall = recall_score(y_test, y_pred_log)          # True positives / All actual positives
    log_f1 = f1_score(y_test, y_pred_log)                  # Harmonic mean of precision and recall

    print("\n📊 Logistic Regression Results:")
    print(f"Accuracy:  {log_accuracy:.3f}")
    print(f"Precision: {log_precision:.3f}")
    print(f"Recall:    {log_recall:.3f}")
    print(f"F1-Score:  {log_f1:.3f}")

    # ========== Random Forest Training ==========
    print("\n🚀 Training Random Forest...")
    start_time : float = time.time()

    # Create the random forest model
    # - n_estimators=100: Number of decision trees in the forest
    # - class_weight='balanced': Handles class imbalance
    # - n_jobs=-1: Use all available CPU cores for parallel processing
    rf_model  : RandomForestClassifier = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # Train the model on training data
    rf_model.fit(X_train, y_train)

    # Record training time
    rf_time = time.time() - start_time
    print(f"✅ Training completed in {rf_time:.2f} seconds")

    # Make predictions on test set
    y_pred_rf = rf_model.predict(X_test)

    # Calculate performance metrics
    rf_accuracy  : float = accuracy_score(y_test, y_pred_rf)
    rf_precision : float = precision_score(y_test, y_pred_rf)
    rf_recall    : float = recall_score(y_test, y_pred_rf)
    rf_f1        : float = f1_score(y_test, y_pred_rf)

    print("\n📊 Random Forest Results:")
    print(f"Accuracy:  {rf_accuracy:.3f}")
    print(f"Precision: {rf_precision:.3f}")
    print(f"Recall:    {rf_recall:.3f}")
    print(f"F1-Score:  {rf_f1:.3f}")

    # ========== Save Models ==========
    # Save trained models as pickle files for later use
    joblib.dump(log_reg, 'models/logistic_model.pkl')
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    print("\n💾 Models saved to /models folder")

    # ========== Generate Comparison Report ==========
    # Write detailed comparison to text file
    with open('results/model_comparison.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("BASELINE MODEL COMPARISON\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("LOGISTIC REGRESSION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training Time: {log_reg_time:.2f} seconds\n")
        f.write(f"Accuracy:      {log_accuracy:.4f}\n")
        f.write(f"Precision:     {log_precision:.4f}\n")
        f.write(f"Recall:        {log_recall:.4f}\n")
        f.write(f"F1-Score:      {log_f1:.4f}\n\n")
        
        f.write("RANDOM FOREST\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training Time: {rf_time:.2f} seconds\n")
        f.write(f"Accuracy:      {rf_accuracy:.4f}\n")
        f.write(f"Precision:     {rf_precision:.4f}\n")
        f.write(f"Recall:        {rf_recall:.4f}\n")
        f.write(f"F1-Score:      {rf_f1:.4f}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("WINNER: ")
        # Determine best model based on F1-score (balance of precision and recall)
        if rf_f1 > log_f1:
            f.write("Random Forest (Better F1-Score)\n")
        else:
            f.write("Logistic Regression (Better F1-Score)\n")

    print("✅ Comparison saved to results/model_comparison.txt")


def main():
    """
    Main execution function for baseline model training pipeline.
    
    This function handles the complete workflow:
    1. Load pre-split training and validation datasets
    2. Drop non-predictive and target columns
    3. Prepare feature matrices and target vectors
    4. Train and evaluate baseline models
    
    The function expects the following file structure:
        data/processed/train_processed.csv
        data/processed/val_processed.csv
    
    And creates output in:
        models/logistic_model.pkl
        models/random_forest_model.pkl
        results/model_comparison.txt
    
    Notes
    -----
    Columns dropped:
    - 'target': The prediction target (cannot be used as a feature)
    - 'unit_id': Equipment identifier (non-predictive)
    - 'source_file': Data source metadata (non-predictive)
    - 'RUL': Remaining Useful Life (would leak future information)
    - 'time_cycles': Temporal index (non-predictive, sequential identifier)
    """
    # Load pre-split training and validation datasets
    train_df  : pd.DataFrame = pd.read_csv('data/processed/train_processed.csv')
    val_df : pd.DataFrame = pd.read_csv('data/processed/val_processed.csv')
    
    # Define columns to exclude from features
    # These are either non-predictive, identifiers, or leak future information
    columns_to_drop  : list = ['target', 'unit_id', 'source_file', 'RUL', 'time_cycles']
    
    # Prepare feature matrices (X) and target vectors (y)
    X_train  : pd.DataFrame = train_df.drop(columns_to_drop, axis=1)
    y_train  : pd.Series = train_df['target']
    
    X_test  : pd.DataFrame = val_df.drop(columns_to_drop, axis=1)
    y_test  : pd.Series = val_df['target']
    
    # Display dataset information
    print(f"✅ Train: {len(X_train)} samples")
    print(f"✅ Val: {len(X_test)} samples")
    print(f"✅ Features: {X_train.shape[1]} columns")
    
    # Train and evaluate baseline models
    train_baseline_models(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()