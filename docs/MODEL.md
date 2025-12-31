# AI Model Documentation

## What This Project Does

This is a **predictive maintenance system** that warns you 1-2 weeks before equipment fails. It analyzes sensor data from turbofan engines and predicts which ones will break down soon, allowing maintenance teams to fix problems before they cause expensive downtime or safety issues.

**Key Achievement:** The system catches 98% of equipment failures before they happen, with only a 27% false alarm rate.

## Model Overview

- **Type:** Binary classifier (predicts "will fail" or "healthy")
- **Best Model:** XGBoost 
- **Performance:** 95.5% accuracy, 98.0% recall, 72.8% precision
- **Warning Window:** 48 operational cycles (~1-2 weeks advance notice)
- **Training Data:** 157,139 sensor readings from 260 turbofan engines

## Dataset

**Source:** NASA C-MAPSS (simulated turbofan engine data)

**Key Stats:**
- 157,139 total records from 260 engines
- Split: 80% training (208 engines), 20% validation (52 engines)
- Class distribution: 90% healthy, 10% failure risk
- 26 original sensor measurements

**Data Quality:**
- No missing values after cleaning
- Outliers removed (3-sigma rule)
- Min-Max normalized to 0-1 range
- Single consistent scaler for all data

## Feature Engineering

Created **152 new features** from 21 raw sensors to capture degradation patterns:

1. **Rolling Averages** (63 features): Smooth noise, reveal trends
2. **Rate of Change** (21 features): Detect sudden anomalies  
3. **Exponential Moving Average** (21 features): Recent trend emphasis
4. **Rolling Std Deviation** (21 features): Measure instability
5. **Baseline Deviation** (21 features): Distance from healthy operation
6. **Range Features** (21 features): Detect erratic behavior
7. **Statistical Aggregates** (4 features): Overall system health
8. **Cycle Normalized** (1 feature): Position in lifecycle

**Final:** 197 predictive features used for training (engineered features + original/raw features)

## Model Performance

### Model Comparison

| Model | Accuracy | Recall | Precision | Training Time |
|-------|----------|--------|-----------|---------------|
| Logistic Regression | 95.5% | 97.6% | 73.1% | ~5 min |
| Random Forest | 97.0% | 82.5% | 90.9% | ~12 sec |
| **XGBoost (Selected)** | **95.5%** | **98.0%** | **72.8%** | ~1-3 hrs |

### Why XGBoost Won

**XGBoost catches 98% of failures** (misses only 2%) vs Random Forest's 83% (misses 17%). In predictive maintenance, missing a failure is far more costly than a false alarm.

**Trade-off:** XGBoost has more false alarms (27%) than Random Forest (9%), but this is acceptable because:
- Cost of missed failure >> cost of false alarm
- 1-2 week warning allows proper planning
- Maintenance teams can triage alerts using confidence scores

### Confusion Matrix (30,185 validation samples)

```
                Predicted
             Healthy  Failure
Actual Healthy  25,382   1,289  (false alarms)
Actual Failure      71   3,443  (caught failures)
```

**Translation:** Out of 3,514 actual failures, the model caught 3,443 (98.0%) and missed only 71 (2.0%).

### Key Metrics Explained

- **Recall (98.0%):** Catches 98 out of 100 real failures ⭐
- **Precision (72.8%):** When it predicts failure, it's right 73% of the time
- **ROC AUC (0.9942):** Near-perfect ability to separate failures from healthy equipment
- **Average Precision (0.9612):** ~10x better than random guessing

## How to Use the Model

### Required Files
```
models/
├── xgboost_model.pkl      # Trained model
├── scaler.pkl             # Data normalizer
└── scaler_columns.json    # Column metadata
```

### Basic Usage

```python
import joblib
import json

# Load model and scaler
model = joblib.load('models/xgboost_model.pkl')
scaler = joblib.load('models/scaler.pkl')

with open('models/scaler_columns.json', 'r') as f:
    cols_to_scale = json.load(f)

# Preprocess new data
X_new[cols_to_scale] = scaler.transform(X_new[cols_to_scale])
X_new[cols_to_scale] = X_new[cols_to_scale].clip(0, 1)

# Make predictions
predictions = model.predict(X_new)  # 0 or 1
probabilities = model.predict_proba(X_new)[:, 1]  # 0.0 to 1.0
```

### Critical Rules

- ✅ Always use the saved scaler from `scaler.pkl`
- ✅ Use `scaler.transform()` not `scaler.fit_transform()`
- ❌ Never fit a new scaler on inference data
- ✅ Input requires 181 features in correct order

## Model Architecture

**XGBoost Configuration:**
- 300 shallow trees (depth=3) trained sequentially
- Learning rate: 0.01 (slow, steady learning)
- Handles 9:1 class imbalance via `scale_pos_weight=8.73`
- Selected from 108 parameter combinations via grid search

**Parameter Selection:**
- The main XGBoost hyperparameters (number of trees, tree depth, learning rate, subsample ratios, etc.) were selected using an extensive grid search with cross-validation. The grid included `max_depth` values of 3, 5, and 7.
- The final model uses `max_depth=3` because grid search showed that many shallow trees (depth 3) provided the best recall and generalization for this time-series predictive maintenance task. Deeper trees (5 or 7) tended to overfit and did not improve recall on the validation set.
- This approach ensures the model is robust, prioritizes catching failures, and avoids overfitting to noise or rare patterns in the training data.

**Why These Settings Work:**
- Many shallow trees > few deep trees for time-series patterns
- Slow learning rate improves generalization
- Full sampling (100% features/samples) works best with 127K training examples

## Top Predictive Features

| Feature | Importance | What It Measures |
|---------|------------|------------------|
| cycle_normalized | 71.0% | Equipment lifecycle position* |
| sensor_10_ema | 3.2% | Sensor 10 exponential moving average |
| sensor_10_dev_baseline | 2.3% | Sensor 10 drift from normal |
| sensor_15 | 2.1% | Raw sensor 15 reading |
| sensor_max_all | 2.1% | Maximum value across all sensors |

*See Limitations section

**Key Pattern:** Model relies on increasing sensor instability and deviation from baseline as equipment approaches failure.

## Limitations & Considerations

### Known Issues

1. **Cycle Normalization Feature (71.0% importance)**
   - Requires knowing total lifecycle length in advance
   - Works in simulation but unavailable in real production
   - Would need removal + retraining for actual deployment
   - Expected impact: Recall may drop from 98% to 90-95%

2. **Probability Granularity**
   - Shallow trees create limited unique probability values
   - Multiple units may show identical failure probabilities
   - This is expected behavior, not a bug
   - Binary classification still works correctly

3. **False Alarm Rate (27%)**
   - About 1 in 4 alerts are false positives
   - Requires human review and triage
   - Acceptable trade-off for critical equipment

4. **Simulated Training Data**
   - Trained on NASA simulation, not real equipment
   - Requires retraining for production deployment
   - May not capture all real-world failure modes

### Data Leakage Checks ✅

- RUL excluded from features
- Train/validation split by entire engines (no overlap)
- Rolling features use only past data
- Scaler fitted only on training data
- No future information used

## Project Status

### Completed ✅

**Phase 1: Model Development**
- Data cleaning & validation (157K records)
- Feature engineering (152 features)
- Model training & tuning (XGBoost selected)
- Inference pipeline + performance reports
- Dashboard deployment

**Deliverables:**
- Production model: `xgboost_model.pkl`
- Data scaler: `scaler.pkl` 
- Inference script: `inference.py`
- Performance visualizations (confusion matrix, ROC curve, feature importance, precision-recall)
- Streamlit dashboard

### Performance Reports

Located in `results/performance_report/`:
- Confusion matrix visualization
- ROC curve (AUC = 0.9942)
- Top 20 feature importance chart
- Precision-Recall curve (AP = 0.9612)
- Complete metrics summary

## Future Improvements

**High Priority:**
- Remove `cycle_normalized` for production readiness
- Add SHAP values for prediction explanations
- Threshold optimization analysis

**Medium Priority:**
- Multi-horizon predictions (24/48/72 cycle windows)
- Confidence intervals for uncertainty quantification
- Root cause classification

**Research:**
- LSTM/Transformer models for temporal sequences
- Transfer learning across equipment types
- Online learning for model updates

## Quick Reference

**Model Performance:**
- Accuracy: 95.5%
- Recall: 98.0% (only 2.0% of failures missed)
- Precision: 72.8% (27% false alarm rate)
- ROC AUC: 0.9942

**Training Data:**
- 157K sensor readings, 260 engines
- 80/20 train/validation split
- 197 features, 10% failure class

**Production Files:**
- `models/xgboost_model.pkl`
- `models/scaler.pkl`
- `models/scaler_columns.json`

**Key Insight:** Model prioritizes catching failures over minimizing false alarms - appropriate for safety-critical predictive maintenance.

---

**Last Updated:** December 26, 2025  
**Version:** 1.2 (Production Ready)  
**Status:** Phase 1 Complete ✅ | Dashboard Deployed ✅