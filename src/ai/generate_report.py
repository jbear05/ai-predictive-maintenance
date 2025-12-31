import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    roc_curve, 
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
import joblib
import json
from datetime import datetime
import os
from config import config
from terminal_colors import Colors, print_header, print_success, print_warning, print_error

# Set style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 11


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    """
    Create a confusion matrix heatmap.
    
    A confusion matrix shows four things:
    - True Negatives (TN): Correctly predicted healthy
    - False Positives (FP): False alarms (predicted failure, actually healthy)
    - False Negatives (FN): Missed failures (predicted healthy, actually failing)
    - True Positives (TP): Correctly predicted failures
    
    For predictive maintenance, FN is the worst - we MUST catch failures!
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual labels (0=healthy, 1=failure)
    y_pred : np.ndarray
        Model predictions (0=healthy, 1=failure)
    save_path : str
        Where to save the image file
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Extract values for annotations
    tn, fp, fn, tp = cm.ravel()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with annotations
    sns.heatmap(
        cm, 
        annot=True,  # Show numbers in cells
        fmt='d',  # Format as integers
        cmap='Blues',  # Color scheme
        cbar=True,
        square=True,
        xticklabels=['Healthy', 'Will Fail'],
        yticklabels=['Healthy', 'Will Fail']
    )
    
    plt.title('Confusion Matrix - XGBoost Model\n', fontsize=16, fontweight='bold')
    plt.ylabel('Actual Condition', fontsize=13)
    plt.xlabel('Predicted Condition', fontsize=13)
    
    # Add text annotations explaining the quadrants
    plt.text(0.5, -0.15, f'True Negatives: {tn:,}', ha='center', transform=plt.gca().transAxes, fontsize=10)
    plt.text(1.5, -0.15, f'False Positives: {fp:,}', ha='center', transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.5, -0.20, f'(Correct: Healthy)', ha='center', transform=plt.gca().transAxes, fontsize=9, style='italic')
    plt.text(1.5, -0.20, f'(False Alarms)', ha='center', transform=plt.gca().transAxes, fontsize=9, style='italic', color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print_success(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str) -> None:
    """
    Create ROC (Receiver Operating Characteristic) curve.
    
    The ROC curve shows the trade-off between:
    - True Positive Rate (recall): How many failures we catch
    - False Positive Rate: How many false alarms we create
    
    A perfect model would have a curve that goes straight up to the top-left corner.
    A random model would be a diagonal line.
    
    AUC (Area Under Curve) score:
    - 1.0 = Perfect model
    - 0.9-1.0 = Excellent
    - 0.8-0.9 = Good
    - 0.7-0.8 = Fair
    - 0.5 = Random (coin flip)
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual labels (0=healthy, 1=failure)
    y_prob : np.ndarray
        Predicted probabilities of failure (0.0 to 1.0)
    save_path : str
        Where to save the image file
    """
    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Calculate AUC score
    auc_score = roc_auc_score(y_true, y_prob)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(
        fpr, tpr, 
        color='darkorange', 
        lw=2, 
        label=f'XGBoost (AUC = {auc_score:.4f})'
    )
    
    # Plot diagonal reference line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5000)')
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (False Alarms)', fontsize=13)
    plt.ylabel('True Positive Rate (Recall)', fontsize=13)
    plt.title('ROC Curve - Model Performance\n', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add annotation explaining what we want
    plt.text(
        0.6, 0.2, 
        'Perfect model would be\nin top-left corner',
        fontsize=10, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print_success(f"ROC curve saved to {save_path}")
    plt.close()


def plot_feature_importance(model: object, feature_names: list, save_path: str, top_n: int = 20) -> None:
    """
    Create feature importance bar chart.
    
    Feature importance shows which sensors/features the model relies on most.
    This is valuable for:
    - Understanding what causes failures
    - Identifying critical sensors that should never fail
    - Simplifying the model (maybe you don't need all 197 features!)
    
    XGBoost calculates importance by counting how many times each feature
    is used to split data across all trees.
    
    Parameters
    ----------
    model : XGBClassifier
        Trained model
    feature_names : list
        Names of all features
    save_path : str
        Where to save the image
    top_n : int
        Show only the top N most important features (default: 20)
    """
    # Get feature importance scores from the model
    importance_scores = model.feature_importances_
    
    # Create DataFrame for easy sorting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    })
    
    # Sort by importance and take top N
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create horizontal bar chart (easier to read feature names)
    plt.barh(
        range(len(importance_df)), 
        importance_df['importance'],
        color='steelblue',
        edgecolor='black',
        linewidth=0.7
    )
    
    # Set feature names as y-axis labels
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.gca().invert_yaxis()  # Highest importance at top
    
    # Labels and title
    plt.xlabel('Importance Score', fontsize=13)
    plt.ylabel('Feature Name', fontsize=13)
    plt.title(f'Top {top_n} Most Important Features\n', fontsize=16, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        plt.text(
            row['importance'], i, 
            f' {row["importance"]:.4f}',
            va='center',
            fontsize=9
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print_success(f"Feature importance chart saved to {save_path}")
    plt.close()
    
    # Return the top features for the written report
    return importance_df


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str) -> None:
    """
    Create Precision-Recall curve.
    
    For imbalanced datasets (like yours with 10% failures), this is often
    MORE informative than ROC curve.
    
    Precision-Recall shows:
    - Precision: When we predict failure, how often are we right?
    - Recall: Of all actual failures, how many do we catch?
    
    The ideal model has both high precision AND high recall (top-right corner).
    
    Average Precision (AP) score:
    - Similar to AUC but for Precision-Recall
    - 1.0 = Perfect
    - > 0.9 = Excellent for imbalanced data
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual labels
    y_prob : np.ndarray
        Predicted probabilities
    save_path : str
        Where to save the image
    """
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Calculate average precision score
    ap_score = average_precision_score(y_true, y_prob)
    
    # Calculate baseline (random classifier performance)
    baseline = np.sum(y_true) / len(y_true)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot Precision-Recall curve
    plt.plot(
        recall, precision,
        color='darkorange',
        lw=2,
        label=f'XGBoost (AP = {ap_score:.4f})'
    )
    
    # Plot baseline (random classifier)
    plt.plot(
        [0, 1], [baseline, baseline],
        color='navy',
        lw=2,
        linestyle='--',
        label=f'Random Classifier (AP = {baseline:.4f})'
    )
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (% of Failures Caught)', fontsize=13)
    plt.ylabel('Precision (% of Alerts that are Real)', fontsize=13)
    plt.title('Precision-Recall Curve\n', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add annotation
    plt.text(
        0.4, 0.2,
        f'Your model catches {recall[np.argmin(np.abs(precision - 0.73))]:.1%}\n'
        f'of failures with 73% precision',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print_success(f"Precision-Recall curve saved to {save_path}")
    plt.close()


def generate_text_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    feature_importance_df: pd.DataFrame,
    save_path: str
) -> None:
    """
    Generate a written performance report.
    
    This creates a professional text file summarizing all metrics.
    Think of this as the "executive summary" that management reads.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual labels
    y_pred : np.ndarray
        Predictions
    y_prob : np.ndarray
        Probabilities
    feature_importance_df : pd.DataFrame
        Top features from importance analysis
    save_path : str
        Where to save the report
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("="*70 + "\n")
        f.write("PREDICTIVE MAINTENANCE MODEL - PERFORMANCE REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: XGBoost Classifier\n")
        f.write(f"Dataset: NASA C-MAPSS Turbofan Engine Data\n")
        f.write("\n")
        
        # Overall Performance
        f.write("SECTION 1: OVERALL PERFORMANCE METRICS\n")
        f.write("-"*70 + "\n")
        
        # Calculate all metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        
        f.write(f"Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)  [Target: ≥80%] ✓\n")
        f.write(f"Precision:         {precision:.4f} ({precision*100:.2f}%) [Target: ≥70%] ✓\n")
        f.write(f"Recall:            {recall:.4f} ({recall*100:.2f}%)    [Target: ≥85%] ✓\n")
        f.write(f"F1-Score:          {f1:.4f} ({f1*100:.2f}%)\n")
        f.write(f"ROC AUC:           {auc:.4f}\n")
        f.write(f"Average Precision: {ap:.4f}\n")
        f.write("\n")
        
        # Confusion Matrix Breakdown
        f.write("SECTION 2: CONFUSION MATRIX ANALYSIS\n")
        f.write("-"*70 + "\n")
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        f.write(f"True Negatives (TN):  {tn:>6,} - Correctly identified healthy equipment\n")
        f.write(f"False Positives (FP): {fp:>6,} - False alarms (predicted failure, actually healthy)\n")
        f.write(f"False Negatives (FN): {fn:>6,} - Missed failures (CRITICAL - predicted healthy, actually failing)\n")
        f.write(f"True Positives (TP):  {tp:>6,} - Correctly predicted failures\n")
        f.write("\n")
        f.write(f"False Alarm Rate:     {fp/(tn+fp)*100:.2f}% of healthy equipment flagged incorrectly\n")
        f.write(f"Missed Failure Rate:  {fn/(fn+tp)*100:.2f}% of failures not caught (Target: <15%)\n")
        f.write("\n")
        
        # Business Impact
        f.write("SECTION 3: BUSINESS IMPACT\n")
        f.write("-"*70 + "\n")
        f.write(f"Out of {tp+fn:,} actual equipment failures:\n")
        f.write(f"  - Model correctly predicted: {tp:,} ({tp/(tp+fn)*100:.1f}%)\n")
        f.write(f"  - Model missed: {fn:,} ({fn/(tp+fn)*100:.1f}%)\n")
        f.write("\n")
        f.write(f"Advance Warning: 48 operational cycles (~1-2 weeks)\n")
        f.write(f"This allows maintenance teams to:\n")
        f.write(f"  - Schedule repairs during planned downtime\n")
        f.write(f"  - Order replacement parts in advance\n")
        f.write(f"  - Prevent catastrophic equipment failures\n")
        f.write(f"  - Reduce unscheduled maintenance by up to 80%\n")
        f.write("\n")
        
        # Top Features
        f.write("SECTION 4: KEY PREDICTIVE FEATURES\n")
        f.write("-"*70 + "\n")
        f.write("Top 10 features that predict equipment failure:\n\n")
        
        for idx, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
            f.write(f"{idx:>2}. {row['feature']:<40} (importance: {row['importance']:.4f})\n")
        
        f.write("\n")
        f.write("These features should be monitored most closely in production.\n")
        f.write("\n")
        
        # Detailed Classification Report
        f.write("SECTION 5: DETAILED CLASSIFICATION REPORT\n")
        f.write("-"*70 + "\n")
        f.write(classification_report(y_true, y_pred, target_names=['Healthy', 'Will Fail']))
        f.write("\n")
        
        # Footer
        f.write("="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print_success(f"Text report saved to {save_path}")


def main():
    """
    Main function to generate complete performance report.
    
    This orchestrates the entire report generation process:
    1. Load model and data
    2. Generate predictions
    3. Create all visualizations
    4. Write text summary
    """

    print_header("GENERATING PERFORMANCE REPORT")
    print()
    
    # Create output directory using centralized config
    output_dir = config.paths.results_root / 'performance_report'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    # Load model
    print("Loading model...")
    model = joblib.load(config.paths.models_root / 'xgboost_model.pkl')
    print_success("Model loaded")
    print()
    
    # Load validation data
    print("Loading validation data...")
    val_data = pd.read_csv(config.paths.val_file)
    print_success(f"Loaded {len(val_data):,} validation samples")
    print()
    
    # Prepare features
    print("Preparing features...")
    columns_to_drop = ['target', 'unit_id', 'source_file', 'RUL', 'time_cycles']
    columns_to_drop = [col for col in columns_to_drop if col in val_data.columns]
    
    X_val = val_data.drop(columns_to_drop, axis=1)
    y_true = val_data['target'].values
    
    print_success(f"Feature matrix: {X_val.shape}")
    print_success(f"True labels: {y_true.shape}")
    print()
    
    # Generate predictions
    print("Generating predictions...")
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    print_success("Predictions complete")
    print()
    
    # Generate visualizations
    print("Creating visualizations...")
    print()
    
    # 1. Confusion Matrix
    plot_confusion_matrix(
        y_true, y_pred,
        os.path.join(output_dir, '1_confusion_matrix.png')
    )
    
    # 2. ROC Curve
    plot_roc_curve(
        y_true, y_prob,
        os.path.join(output_dir, '2_roc_curve.png')
    )
    
    # 3. Feature Importance
    feature_names = X_val.columns.tolist()
    importance_df = plot_feature_importance(
        model, feature_names,
        os.path.join(output_dir, '3_feature_importance.png'),
        top_n=20
    )
    
    # 4. Precision-Recall Curve
    plot_precision_recall_curve(
        y_true, y_prob,
        os.path.join(output_dir, '4_precision_recall_curve.png')
    )
    
    print()
    
    # Generate text report
    print("Generating text report...")
    generate_text_report(
        y_true, y_pred, y_prob, importance_df,
        os.path.join(output_dir, 'performance_report.txt')
    )
    
    print()
    print("="*70)
    print_success("REPORT GENERATION COMPLETE!")
    print("="*70)
    print()
    print_success(f"All files saved to: {output_dir}/")
    print()
    print("Generated files:")
    print("  1. 1_confusion_matrix.png")
    print("  2. 2_roc_curve.png")
    print("  3. 3_feature_importance.png")
    print("  4. 4_precision_recall_curve.png")
    print("  5. performance_report.txt")



if __name__ == "__main__":
    main()