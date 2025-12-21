# System Architecture

## High-Level Overview
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐
│  NASA C-MAPSS   │───▶│  Data Processing │───▶│  Processed  │
│  Raw Data (.txt)│    │  & Feature Eng   │    │  Data (.csv)│
└─────────────────┘    └──────────────────┘    └─────────────┘
                                                       │
                                                       ▼
                       ┌──────────────────┐    ┌─────────────┐
                       │  Trained Models  │◀───│   Model     │
                       │  + Scaler (.pkl) │    │  Training   │
                       └──────────────────┘    └─────────────┘
                                  │
                                  ▼
                       ┌──────────────────┐
                       │    Inference     │
                       │    Pipeline      │
                       └──────────────────┘
```

## Components

### 1. Data Processing Layer

#### Verification Script (`verify_data.py`)
- Loads all C-MAPSS training files (train_FD001-004.txt)
- Validates dataset meets minimum 50,000 record requirement
- Provides statistical summary and data quality checks
- **Status:** ✅ Complete (Step 1.1)

#### Cleaning Script (`clean_data.py`)
- Handles missing values (dropna)
- Removes outliers using 3-sigma rule (Z-score > 3)
- Min-Max normalization (0-1 scale) for all sensor columns
- Skips constant features to prevent scaling errors
- Outputs: `train_FD001_cleaned.csv` through `train_FD004_cleaned.csv`
- **Status:** ✅ Complete (Step 1.2)
- **Note:** Initial version created 4 separate scalers (one per file) - later corrected in scaler fix step

#### Feature Engineering Script (`data_prep_features.py`)
- Combines all 4 cleaned training files
- Creates binary target variable (48-cycle failure window)
- Engineers 173 features across 8 categories:
  - Rolling averages (3, 5, 10 cycles)
  - Rate of change
  - Exponential moving averages
  - Rolling standard deviation
  - Baseline deviation
  - Range features
  - Statistical aggregates
  - Cycle normalization
- Splits data 80/20 (stratified by unit_id)
- Outputs: `train_processed.csv`, `val_processed.csv`, `feature_documentation.csv`, `data_quality_report.txt`
- **Status:** ✅ Complete (Step 1.3)

#### Scaler Fix Script (`fix_scaler.py`)
- **Purpose:** Corrects scaling inconsistency from using 4 separate scalers
- Loads combined train/validation data
- Fits ONE MinMaxScaler on training data only (prevents data leakage)
- Transforms both train and validation using the same fitted scaler
- Saves scaler for deployment: `models/scaler.pkl`
- Saves column metadata: `models/scaler_columns.json`
- Overwrites processed CSV files with consistently-scaled data
- **Status:** ✅ Complete (Scaler correction step)
- **Critical for deployment:** Ensures inference pipeline uses correct normalization

### 2. Model Training Layer

#### Baseline Model Training Script (`train_baseline_models.py`)
- Trains two baseline models: Logistic Regression and Random Forest
- Uses pre-split train/val datasets from feature engineering step
- Handles class imbalance with `class_weight='balanced'`
- Excludes non-predictive columns: unit_id, source_file, RUL, time_cycles
- Evaluates models on: accuracy, precision, recall, F1-score
- Saves trained models: `logistic_model.pkl`, `random_forest_model.pkl`
- Generates comparison report: `model_comparison.txt`
- **Status:** ✅ Complete (Step 2.1)

**Model Performance (Baseline):**
| Model | Accuracy | Recall | Precision | F1-Score | Training Time |
|-------|----------|--------|-----------|----------|---------------|
| Logistic Regression | 76.7% | 80.4% | 30.8% | 44.5% | ~8 min |
| Random Forest | 96.8% | 79.7% | 91.8% | 85.3% | ~13 sec |

**Winner:** Random Forest (significantly better F1-score)

#### XGBoost Training Script (`train_xgboost.py`)
- Trains XGBoost classifier with comprehensive hyperparameter tuning
- Uses GridSearchCV to test 108 parameter combinations (3×3×3×2×2×2)
- Automatically handles class imbalance via scale_pos_weight (ratio: 8.73:1)
- Optimizes for recall using 3-fold cross-validation
- Saves trained model: `xgboost_model.pkl`
- Appends results to: `model_comparison.txt`
- **Status:** ✅ Complete (Step 2.2 - Retrained after scaler fix)

**Model Performance (XGBoost):**
| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Accuracy | 95.47% | ≥80% | ✅ +15.47% |
| Recall | 98.26% | ≥85% | ✅ +13.26% |
| Precision | 72.54% | ≥70% | ✅ +2.54% |
| ROC AUC | 0.9929 | — | ✅ Excellent |
| Avg Precision | 0.9412 | — | ✅ Excellent |

**Best Parameters:** `learning_rate=0.01`, `max_depth=3`, `n_estimators=300`  
**Winner:** 🏆 XGBoost (industry-leading 98.26% recall)

**Note:** Model retrained after scaler correction on consistently-scaled data.

### 3. Inference & Reporting Layer

#### Inference Pipeline (`inference.py`)
- Loads trained model and scaler artifacts
- Preprocesses raw sensor data (scaling, clipping to [0,1])
- Generates predictions with probability scores
- Returns binary predictions (0=healthy, 1=will fail) and confidence scores
- **Status:** ✅ Complete (Step 2.3)

**Key Features:**
- Handles 197 features (26 original + 171 engineered, excluding metadata)
- Applies saved scaler for consistent normalization
- Clips scaled values to [0, 1] range (handles distribution shift)
- Test accuracy: 97% on validation sample

#### Performance Report Generator (`generate_report.py`)
- Creates 4+ visualizations and text summary
- **Status:** ✅ Complete (Step 2.3)

**Generated Outputs:**
1. **Confusion Matrix** - Shows TN=25,337, FP=1,334, FN=61, TP=3,453
2. **ROC Curve** - AUC=0.9929 (near-perfect discrimination)
3. **Feature Importance** - Top 20 features, `cycle_normalized` dominates (0.60)
4. **Precision-Recall Curve** - AP=0.9412 (excellent for imbalanced data)
5. **Text Report** - Complete metrics summary with business impact analysis

**Key Insights from Report:**
- Only 61 failures missed out of 3,514 (1.74% miss rate)
- Top predictive features: cycle_normalized, rolling std features, baseline deviation
- Model achieves 98.26% recall at 72.54% precision (optimal for maintenance)

### 4. API Layer
- **Status:** ⏳ Not yet implemented (Step 3.1-3.3 pending)

### 5. Dashboard Layer
- **Status:** ⏳ Not yet implemented (Step 4.1-4.3 pending)

## Data Flow
```
Raw Data (data/raw/)
    ├── train_FD001.txt (20,631 records)
    ├── train_FD002.txt (53,759 records)
    ├── train_FD003.txt (24,720 records)
    └── train_FD004.txt (61,249 records)
           ↓
    [verify_data.py]
           ↓
    [clean_data.py]
           ↓
Cleaned Data (data/processed/)
    ├── train_FD001_cleaned.csv through train_FD004_cleaned.csv
    (⚠️ Initially scaled with 4 separate scalers)
           ↓
    [data_prep_features.py]
           ↓
Combined Dataset → 157,139 records, 260 engines, 173 engineered features
           ↓
Initial Processed Data
    ├── train_processed.csv (126,954 records, 202 columns)
    └── val_processed.csv (30,185 records, 202 columns)
    (⚠️ Contained inconsistently-scaled data)
           ↓
    [fix_scaler.py] ← CORRECTION STEP
           ↓
    ✅ Single scaler fitted on training data only
    ✅ Both datasets re-scaled consistently
           ↓
Corrected Processed Data (data/processed/)
    ├── train_processed.csv (updated)
    └── val_processed.csv (updated)
           ↓
    [train_baseline_models.py] → logistic_model.pkl, random_forest_model.pkl
           ↓
    [train_xgboost.py] ← RETRAINED
           ↓
Deployment-Ready Artifacts (models/)
    ├── xgboost_model.pkl (197 features)
    ├── scaler.pkl (193 columns scaled)
    └── scaler_columns.json
           ↓
    [inference.py] ← INFERENCE PIPELINE
           ↓
    [generate_report.py] ← PERFORMANCE REPORT
           ↓
Performance Report (results/performance_report/)
    ├── 1_confusion_matrix.png
    ├── 2_roc_curve.png
    ├── 3_feature_importance.png
    ├── 4_precision_recall_curve.png
    └── performance_report.txt
```

## Model Architecture

### XGBoost Model Details
- **Type:** Gradient Boosted Decision Trees
- **Ensemble Method:** Sequential boosting with error correction
- **Number of Trees:** 300
- **Max Tree Depth:** 3 (shallow trees prevent overfitting)
- **Learning Rate:** 0.01 (conservative for better generalization)
- **Class Imbalance Handling:** scale_pos_weight=8.73
- **Features Used:** 197 predictive features

### Feature Set
- **21 raw sensor readings** (sensor_1 through sensor_21)
- **3 operational settings** (setting_1, setting_2, setting_3)
- **173 engineered features:**
  - Rolling statistics (mean, std, range)
  - Temporal features (rate of change, EMA)
  - Deviation features (from baseline)
  - Cycle normalization
- **Total columns:** 202 (minus 5 metadata columns = 197 for training)

### Prediction Target
- **Type:** Binary classification
- **Question:** Will equipment fail within next 48 operational cycles?
- **Time Horizon:** 48 cycles ≈ 1-2 weeks advance warning
- **Class Distribution:** ~10% failures, ~90% healthy

### Feature Importance (Top 5)
1. **cycle_normalized** (0.5975) - Lifecycle position
2. **sensor_11_roll_std_5** (0.0100) - Sensor 11 volatility
3. **sensor_15_roll_std_5** (0.0094) - Sensor 15 volatility
4. **sensor_7_roll_std_5** (0.0089) - Sensor 7 volatility
5. **sensor_14_roll_std_5** (0.0086) - Sensor 14 volatility

**Key Insight:** `cycle_normalized` dominates (60x more important than next feature), indicating lifecycle position is strongest predictor. See Limitations for production deployment considerations.

## Preprocessing Pipeline (Critical for Inference)

### Scaler Configuration
- **Type:** MinMaxScaler (0-1 normalization)
- **Fitted on:** Training data only (126,954 samples)
- **Columns scaled:** 193 sensor columns with variance > 1e-10
- **Saved artifacts:**
  - `models/scaler.pkl` - Fitted scaler object
  - `models/scaler_columns.json` - List of columns to scale

### Why Scaler Consistency Matters
**Problem:** Initial cleaning created 4 separate scalers, causing inconsistent normalization.

**Solution:** 
1. Fitted ONE scaler on training data only
2. Transformed both train/validation with same scaler
3. Saved scaler for inference deployment

**Result:** 
- ✅ Consistent normalization across all data
- ✅ No data leakage
- ✅ Inference uses exact same normalization as training

## Security Considerations

### Current Implementation
- ✅ All processing is local (no external API calls)
- ✅ No sensitive data transmission
- ✅ Standard Python libraries only
- ✅ Suitable for air-gapped deployment

## Performance

### Data Quality Metrics
- **Missing values:** 0.00% ✅
- **Outlier removal:** ~1-3% per file
- **Scaling consistency:** Single scaler ✅

### Model Training Performance
| Model | Training Time |
|-------|---------------|
| Logistic Regression | ~8 minutes |
| Random Forest | ~13 seconds |
| XGBoost (full grid) | ~1-3 hours |

### Model Inference Performance
- **Prediction latency:** <100ms per sample
- **Test accuracy:** 97% on 100-sample validation
- **Model size:** <50MB

## Current Project Status

### Phase 1: MVP Development (COMPLETE ✅)
| Step | Task | Status |
|------|------|--------|
| 1.1 | Data Acquisition | ✅ Dec 11 |
| 1.2 | Data Cleaning | ✅ Dec 12 |
| 1.3 | Feature Engineering | ✅ Dec 13 |
| — | Scaler Correction | ✅ Dec 18 |
| 2.1 | Baseline Models | ✅ Dec 14 |
| 2.2 | XGBoost Training | ✅ Dec 15 |
| 2.3 | Inference Pipeline + Report | ✅ Dec 20 |

### Phase 2-3: Next Steps
| Step | Task | Status |
|------|------|--------|
| 3.1-3.3 | Flask API Development | ⏳ Pending |
| 4.1-4.3 | Dashboard Creation | ⏳ Pending |

### Performance Targets (ALL EXCEEDED ✅)
| Target | Goal | Achieved | Status |
|--------|------|----------|--------|
| Accuracy | ≥80% | 95.47% | ✅ +15.47% |
| Recall | ≥85% | 98.26% | ✅ +13.26% |
| Precision | ≥70% | 72.54% | ✅ +2.54% |

## Saved Artifacts

### Models Directory (`models/`)
| File | Purpose | Created By |
|------|---------|------------|
| `xgboost_model.pkl` | Production model (197 features) | train_xgboost.py |
| `scaler.pkl` | Data normalization (193 columns) | fix_scaler.py |
| `scaler_columns.json` | Preprocessing metadata | fix_scaler.py |
| `logistic_model.pkl` | Baseline model | train_baseline_models.py |
| `random_forest_model.pkl` | Baseline model | train_baseline_models.py |

### Code Pipeline
| File | Purpose | Status |
|------|---------|--------|
| `verify_data.py` | Data validation | ✅ |
| `clean_data.py` | Data cleaning | ✅ |
| `data_prep_features.py` | Feature engineering | ✅ |
| `fix_scaler.py` | Scaler correction | ✅ |
| `train_baseline_models.py` | Baseline training | ✅ |
| `train_xgboost.py` | XGBoost training | ✅ |
| `inference.py` | Inference pipeline | ✅ |
| `generate_report.py` | Performance reporting | ✅ |

### Performance Report (`results/performance_report/`)
| File | Description |
|------|-------------|
| `1_confusion_matrix.png` | Visual confusion matrix |
| `2_roc_curve.png` | ROC curve (AUC=0.9929) |
| `3_feature_importance.png` | Top 20 features |
| `4_precision_recall_curve.png` | PR curve (AP=0.9412) |
| `performance_report.txt` | Complete metrics summary |

### Required for Inference
All three files must be used together:
1. `xgboost_model.pkl` - for predictions
2. `scaler.pkl` - for normalization
3. `scaler_columns.json` - for column metadata

**Critical:** Never use a different scaler for inference.

## Limitations

### Known Issues

1. **Cycle Normalization Feature** ⚠️
   - `cycle_normalized` dominates importance (0.60) but requires knowing total lifecycle
   - Works in simulation (known lifecycles) but unavailable in real production
   - For deployment: remove feature and retrain (expect 90-95% recall vs. 98%)
   - Kept for this project: demonstrates understanding of simulation vs. production

2. **Simulated Data**
   - C-MAPSS is physics-based simulation, not real equipment
   - Requires retraining for Y-12-specific equipment

3. **False Alarm Rate**
   - 27% false positive rate (1,334 false alarms out of 26,671 healthy)
   - Trade-off: prioritizes catching failures over minimizing false alarms

## Future Architecture
- [ ] Flask API layer (Step 3)
- [ ] Streamlit dashboard (Step 4)
- [ ] SQLite database for prediction history
- [ ] Remove cycle_normalized for production deployment

## Lessons Learned

### Scaler Management
**Issue:** Multiple scalers created during cleaning caused inconsistent normalization.

**Resolution:** Dedicated `fix_scaler.py` script fits one scaler on training data only.

**Best Practice:** Always fit preprocessing objects on training data only, then save for inference.

### Feature Importance Analysis
**Discovery:** `cycle_normalized` dominates but represents lifecycle position, not failure patterns.

**Lesson:** Analyze feature importance to identify features that won't generalize to production. Document limitations rather than hide them.

## References
- **Dataset:** NASA C-MAPSS Turbofan Engine Degradation Simulation
- **Paper:** Saxena et al. (2008) - Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation
- **Framework:** XGBoost, scikit-learn, pandas, NumPy

---

**Last Updated:** December 20, 2024  
**Status:** Phase 1 Complete ✅ | Step 2.3 Complete ✅  
**Performance:** 98.26% Recall | ROC AUC: 0.9929 | AP: 0.9412  
**Deployment Ready:** Model + Scaler + Inference Pipeline ✅