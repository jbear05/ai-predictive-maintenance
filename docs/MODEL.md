# AI Model Documentation

## Model Overview
**Type:** Binary Classification
**Algorithm:** [Not yet trained]
**Target:** Predict if equipment will fail in next 48 hours

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
- Healthy (Class 0): 140,571 records (89.5%)
- Failure Risk (Class 1): 16,568 records (10.5%)
- Failure window: 48 cycles (last 48 cycles of each engine's life marked as failure risk)

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

### Hyperparameter Tuning
[Not yet performed - Step 2.2 pending]

**Final hyperparameters:**
[Not yet trained - Step 2.2 pending]

## Performance Metrics

### Test Set Results
[Not yet trained - Step 2.3 pending]

### Confusion Matrix
[Not yet trained - Step 2.3 pending]

### What This Means
[Not yet trained - Step 2.3 pending]

### Feature Importance
Top 10 most important features:
[Not yet trained - Step 2.3 pending]

## Inference

### How to Use
[Not yet implemented - Step 3 pending]

### Prediction Thresholds
[Not yet determined - Step 2.3 pending]

## Limitations
1. **Training data:** Simulated turbofan data, not actual DOE equipment
2. **Requires retraining:** For deployment on Y-12-specific equipment
3. **Sensor coverage:** Assumes all 21 sensors available

## Future Improvements
1. Add root cause analysis (classify failure type)
2. Implement confidence intervals
3. Add explainability (SHAP values)
4. Online learning (update model with new data)
5. Multi-equipment models (transfer learning)

## References
1. NASA C-MAPSS Dataset: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
2. XGBoost documentation: [pending]
3. Research papers: [pending]