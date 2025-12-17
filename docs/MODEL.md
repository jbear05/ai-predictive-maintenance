# AI Model Documentation

## Model Overview
**Type:** Binary Classification
**Algorithms Trained:** Logistic Regression, Random Forest Classifier
**Best Model:** Random Forest (96.8% accuracy, 85.3% F1-score)
**Target:** Predict if equipment will fail in next 48 hours (binary: 0 = healthy, 1 = failure risk)

## Dataset

**Source:** NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) Dataset

**Files Used:**
- train_FD001.txt (cleaned: 19,337 records)
- train_FD002.txt (cleaned: 53,759 records)
- train_FD003.txt (cleaned: 22,794 records)
- train_FD004.txt (cleaned: 61,249 records)

**Combined Dataset:**
- Total records: 157,139 cycles
- Total units (engines): 260
- Missing values: 0.00%

**Train/Validation Split:**
- Training set: 126,954 records (208 units) - 80%
- Validation set: 30,185 records (52 units) - 20%
- Split method: Stratified by unit_id to prevent data leakage

**Target Distribution:**
- Healthy (Class 0): 113,900 records in training set (89.7%)
- Failure Risk (Class 1): 13,054 records in training set (10.3%)
- Failure window: 48 cycles (last 48 cycles of each engine's life marked as failure risk)
- Class imbalance handled via `class_weight='balanced'` in models

## Feature Engineering

### Raw Sensors
**26 Original Columns:**
- `unit_id`: Engine identifier
- `time_cycles`: Operational cycle number
- `setting_1`, `setting_2`, `setting_3`: Operational settings
- `sensor_1` through `sensor_21`: 21 sensor measurements (temperature, pressure, speed, etc.)

### Engineered Features
**Total: 173 engineered features created**

**Feature Categories:**

1. **Rolling Averages (63 features)**
   - 3-cycle, 5-cycle, and 10-cycle rolling averages for each of 21 sensors
   - Purpose: Smooth noise and reveal trends
   - Example: `sensor_1_roll_avg_3`, `sensor_14_roll_avg_10`

2. **Rate of Change (21 features)**
   - First difference (current value - previous value) for each sensor
   - Purpose: Detect sudden changes or anomalies
   - Example: `sensor_7_rate_change`

3. **Exponential Moving Average (21 features)**
   - Weighted average emphasizing recent values for each sensor
   - Span: 5 cycles
   - Purpose: Responsive trend detection
   - Example: `sensor_11_ema`

4. **Rolling Standard Deviation (21 features)**
   - 5-cycle rolling standard deviation for each sensor
   - Purpose: Measure volatility and instability
   - Example: `sensor_14_roll_std_5`

5. **Baseline Deviation (21 features)**
   - Difference from healthy baseline (first 20% of each engine's lifecycle)
   - Purpose: Measure degradation from normal operation
   - Example: `sensor_2_dev_baseline`

6. **Range Features (21 features)**
   - Max - Min over last 5 cycles for each sensor
   - Purpose: Detect erratic behavior
   - Example: `sensor_9_range_5`

7. **Statistical Aggregates (4 features)**
   - `sensor_mean`: Average across all 21 sensors at each time step
   - `sensor_std`: Standard deviation across all 21 sensors
   - `sensor_max`: Maximum sensor value at each time step
   - `sensor_min`: Minimum sensor value at each time step
   - Purpose: Overall system health indicators

8. **Cycle Normalized (1 feature)**
   - `cycle_normalized`: Current cycle / max cycle for that engine (0.0 to 1.0)
   - Purpose: Lifecycle context awareness

**Final Dataset Shape:**
- 202 total columns (26 original + 3 derived + 173 engineered)
- 194 features used for training (excluded: target, unit_id, source_file, RUL, time_cycles)
- Training samples: 101,563
- Validation samples: 25,391

## Model Training

### Preprocessing Steps
**Data Cleaning (Step 1.2):**
- Removed null values (final: 0.00% missing)
- Removed outliers using 3-sigma rule (Z-score > 3)
- Min-Max normalization applied to all sensor columns (scaled to 0-1 range)
- Skipped constant features (zero variance) to prevent scaling errors

**Feature Engineering (Step 1.3):**
- Created 173 engineered features across 8 categories
- Processed each engine separately to maintain temporal order
- Computed rolling statistics with appropriate window sizes
- Generated target variable with 48-cycle failure window

**Feature Selection (Step 2.1):**
- Excluded non-predictive columns: unit_id, source_file, time_cycles
- Excluded RUL (Remaining Useful Life) to prevent data leakage
- Final feature count: 194 predictive features

### Baseline Model Training (Step 2.1 - Complete ✅)

**Models Trained:**

1. **Logistic Regression**
   - Solver: SAGA (stochastic average gradient descent)
   - Max iterations: 2,000
   - Class weight: Balanced (to handle 10.3% minority class)
   - Random state: 42 (for reproducibility)
   - Training time: 483.71 seconds (~8 minutes)

2. **Random Forest Classifier**
   - Number of estimators: 100 trees
   - Class weight: Balanced
   - n_jobs: -1 (parallel processing with all CPU cores)
   - Random state: 42
   - Training time: 13.22 seconds

**Model Selection Criteria:**
- Primary metric: F1-score (balance of precision and recall)
- Secondary metrics: Accuracy, precision, recall
- Recall priority: Critical to catch actual failures (minimize false negatives)

### Hyperparameter Tuning
**Baseline Models:** Used default/standard hyperparameters
[XGBoost hyperparameter tuning pending - Step 2.2]

**XGBoost hyperparameters:**
[Not yet trained - Step 2.2 pending]

## Performance Metrics

### Baseline Model Results (Validation Set)

**Logistic Regression:**
- Accuracy: 76.7%
- Precision: 30.8% (many false positives)
- Recall: 80.4% (catches most failures)
- F1-Score: 44.5%
- Training time: 483.71 seconds

**Random Forest (BEST BASELINE):**
- Accuracy: 96.8% ⭐
- Precision: 91.8% (reliable predictions)
- Recall: 79.7% (catches most failures)
- F1-Score: 85.3% ⭐
- Training time: 13.22 seconds

**Winner:** Random Forest (significantly better F1-score and accuracy)

### Model Comparison Analysis

**Why Random Forest Outperforms:**
1. Captures non-linear relationships in engineered features
2. Handles complex interactions between sensors
3. Naturally robust to outliers
4. Effective with rolling averages and time-series features

**Logistic Regression Limitations:**
1. Linear decision boundary insufficient for complex patterns
2. High false positive rate (70% of "failure" predictions incorrect)
3. Cannot capture feature interactions effectively

**Trade-offs:**
- Random Forest: Better accuracy, slower inference (100 trees)
- Logistic Regression: Faster inference, lower accuracy, more false alarms

### XGBoost Results
[Not yet trained - Step 2.2 pending]

### Confusion Matrix
[Not yet trained - Step 2.3 pending]

### What This Means

**For Random Forest (Current Best Model):**

**Accuracy (96.8%):** Out of 100 predictions, 97 are correct
- Very reliable overall performance

**Precision (91.8%):** When model predicts "WILL FAIL":
- 92 out of 100 predictions are correct
- Only 8 false alarms per 100 alerts
- Maintenance teams can trust these alerts

**Recall (79.7%):** Of actual failures:
- Model catches about 80 out of 100 real failures
- Misses about 20 out of 100 (false negatives)
- **Room for improvement** - we want to catch more failures

**F1-Score (85.3%):** 
- Strong balance between precision and recall
- Indicates robust model performance

**Business Impact:**
- High precision = fewer unnecessary maintenance actions
- Good recall = catches most equipment failures before they happen
- 20% missed failures = potential for improvement with XGBoost

### Feature Importance
Top 10 most important features for Random Forest:
[Analysis pending - will be generated in Step 2.3 with visualization]

**Expected Important Features (based on domain knowledge):**
- Rolling standard deviations (detect increasing instability)
- Baseline deviation features (measure degradation)
- Rate of change features (sudden anomalies)
- Recent rolling averages (current health state)

## Inference

### How to Use
[Not yet implemented - Step 3 pending]

### Prediction Thresholds
**Default threshold:** 0.5 (standard classification threshold)
- Probability ≥ 0.5 → Predict failure risk (Class 1)
- Probability < 0.5 → Predict healthy (Class 0)

[Threshold optimization pending - Step 2.3]
- May adjust threshold to improve recall (catch more failures)
- Trade-off: Higher recall = more false positives
- Business decision: Cost of missed failure vs. unnecessary maintenance

## Limitations
1. **Training data:** Simulated turbofan data, not actual DOE equipment
2. **Requires retraining:** For deployment on Y-12-specific equipment
3. **Sensor coverage:** Assumes all 21 sensors available
4. **Recall limitation:** Current model misses ~20% of failures (79.7% recall)
5. **Class imbalance:** Only 10.3% failure samples may limit pattern learning
6. **No temporal validation:** Models validated on held-out units, not future time periods
7. **RUL dependency:** Original dataset includes RUL (removed to prevent leakage, but limits comparison to published benchmarks)

## Future Improvements
1. **XGBoost model** (Step 2.2) - Target ≥85% recall with GridSearchCV tuning
2. **Threshold optimization** - Adjust prediction threshold to catch more failures
3. **Add root cause analysis** - Classify failure type based on sensor patterns
4. **Implement confidence intervals** - Quantify prediction uncertainty
5. **Add explainability** - SHAP values to explain individual predictions
6. **Online learning** - Update model with new data without full retraining
7. **Multi-equipment models** - Transfer learning across different engine types
8. **Ensemble methods** - Combine multiple models for better predictions
9. **Cost-sensitive learning** - Weight false negatives more heavily than false positives
10. **Feature selection** - Identify and remove redundant engineered features

## References
1. NASA C-MAPSS Dataset: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
2. Scikit-learn documentation: https://scikit-learn.org/
3. Logistic Regression: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
4. Random Forest: https://scikit-learn.org/stable/modules/ensemble.html#forest
5. XGBoost documentation: [pending - Step 2.2]