import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import time


def train_baseline_models(X_train : pd.DataFrame, y_train : pd.Series, X_test : pd.DataFrame, y_test : pd.Series) -> None:
    print("🚀 Starting baseline model training...\n")

    print("\n🚀 Training Logistic Regression...")
    start_time = time.time()

    # Create the model
    log_reg = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=2000,  # Changed from 1000
        solver='saga'   # Different solver
    )

    # Train it
    log_reg.fit(X_train, y_train)

    # How long did it take?
    log_reg_time = time.time() - start_time

    print(f"✅ Training completed in {log_reg_time:.2f} seconds")

    # Make predictions on test set
    y_pred_log = log_reg.predict(X_test)

    # Calculate scores
    log_accuracy = accuracy_score(y_test, y_pred_log)
    log_precision = precision_score(y_test, y_pred_log)
    log_recall = recall_score(y_test, y_pred_log)
    log_f1 = f1_score(y_test, y_pred_log)

    print("\n📊 Logistic Regression Results:")
    print(f"Accuracy:  {log_accuracy:.3f}")
    print(f"Precision: {log_precision:.3f}")
    print(f"Recall:    {log_recall:.3f}")
    print(f"F1-Score:  {log_f1:.3f}")


    print("\n🚀 Training Random Forest...")
    start_time = time.time()

    # Create the model
    rf_model = RandomForestClassifier(
        n_estimators=100,  # Use 100 decision trees
        class_weight='balanced',
        random_state=42,
        n_jobs=-1  # Use all CPU cores (makes it faster!)
    )

    # Train it
    rf_model.fit(X_train, y_train)

    rf_time = time.time() - start_time

    print(f"✅ Training completed in {rf_time:.2f} seconds")

    # Make predictions
    y_pred_rf = rf_model.predict(X_test)

    # Calculate scores
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_precision = precision_score(y_test, y_pred_rf)
    rf_recall = recall_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf)

    print("\n📊 Random Forest Results:")
    print(f"Accuracy:  {rf_accuracy:.3f}")
    print(f"Precision: {rf_precision:.3f}")
    print(f"Recall:    {rf_recall:.3f}")
    print(f"F1-Score:  {rf_f1:.3f}")

    # Save both models
    joblib.dump(log_reg, 'models/logistic_model.pkl')
    joblib.dump(rf_model, 'models/random_forest_model.pkl')

    print("\n💾 Models saved to /models folder")

    # Write results to file
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
        if rf_f1 > log_f1:
            f.write("Random Forest (Better F1-Score)\n")
        else:
            f.write("Logistic Regression (Better F1-Score)\n")

    print("✅ Comparison saved to results/model_comparison.txt")

def main():
    # Load pre-split files
    train_df = pd.read_csv('data/processed/train_processed.csv')
    val_df = pd.read_csv('data/processed/val_processed.csv')
    
    # Drop cheating and non-informative columns
    columns_to_drop = ['target', 'unit_id', 'source_file', 'RUL', 'time_cycles']
    
    X_train = train_df.drop(columns_to_drop, axis=1)
    y_train = train_df['target']
    
    X_test = val_df.drop(columns_to_drop, axis=1)
    y_test = val_df['target']
    
    print(f"✅ Train: {len(X_train)} samples")
    print(f"✅ Val: {len(X_test)} samples")
    print(f"✅ Features: {X_train.shape[1]} columns")
    
    train_baseline_models(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()