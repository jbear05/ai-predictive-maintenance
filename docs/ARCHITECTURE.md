# System Architecture

## What This System Does

A **predictive maintenance pipeline** that processes turbofan engine sensor data, trains machine learning models, and predicts equipment failures 1-2 weeks in advance.

> For model performance details, see [MODEL.md](MODEL.md)

## High-Level Architecture

```
Raw Data → Data Processing → Model Training → Inference Pipeline
  (NASA)    (Clean + Features)   (XGBoost)     (Predictions)
```

## Core Components

### 1. Data Processing Pipeline

**Input:** NASA C-MAPSS dataset (160K records, 260 engines)  
**Output:** Clean, engineered features ready for ML

| Script | Purpose | Key Actions |
|--------|---------|-------------|
| `verify_data.py` | Validate data quality | Check 50K+ records, generate stats |
| `clean_data.py` | Clean & normalize | Remove outliers, scale to 0-1 range |
| `data_prep_features.py` | Feature engineering | Create 152 features, 80/20 split |
| `fix_scaler.py` | Fix normalization | Single scaler for consistency ⭐ |

**Critical Fix Applied:** Original cleaning created 4 separate scalers (one per file), causing inconsistent normalization. Fixed by fitting ONE scaler on training data only and saving it for inference.

### 2. Model Training Pipeline

| Script | Models Trained | Performance |
|--------|----------------|-------------|
| `train_baseline_models.py` | Logistic Regression, Random Forest | 83% recall baseline |
| `train_xgboost.py` | XGBoost (108 configs tested) | 98% recall ⭐ |

**Model Selection:** XGBoost won due to 98% recall vs Random Forest's 83% (catches 15% more failures despite more false alarms).

### 3. Inference & Reporting

| Script | Purpose | Output |
|--------|---------|--------|
| `inference.py` | Production predictions | Binary + probability scores |
| `generate_report.py` | Performance analysis | 4 visualizations + metrics |

### 4. Dashboard

| Script | Purpose |
|--------|---------|
| `app.py` | Main Dash application |
| `charts.py` | Visualization components |
| `results.py` | Results display |
| `risk.py` | Risk assessment logic |
| `state.py` | Application state management |
| `validation.py` | Input validation |

## Data Flow

```
train_FD001-004.txt (160K records)
         ↓
    Clean + Remove Outliers
         ↓
    ⚠️ 4 separate scalers created
         ↓
    Combine + Engineer 152 features
         ↓
    Split 80/20 by engine
         ↓
    ⚠️ Inconsistent scaling detected
         ↓
    fix_scaler.py → ONE scaler fitted on train only
         ↓
    ✅ Consistently scaled data
         ↓
    Train Logistic + Random Forest + XGBoost
         ↓
    ✅ XGBoost selected (98% recall)
         ↓
    Generate performance report
         ↓
Deployment-ready artifacts:
  • xgboost_model.pkl (197 features)
  • scaler.pkl (single normalizer)
  • scaler_columns.json (metadata)
```

## Feature Engineering

**Input:** 26 columns (21 sensors + 3 settings + 2 metadata)  
**Output:** 152 engineered features + original sensors

**Feature Categories (152 created):**
1. Rolling averages (3, 5, 10 cycles) - 63 features
2. Rate of change (first difference) - 21 features
3. Exponential moving average - 21 features
4. Rolling std deviation (volatility) - 21 features
5. Baseline deviation (drift from healthy) - 21 features
6. Statistical aggregates (mean, std, max, min) - 4 features
7. Cycle normalized (lifecycle position) - 1 feature

## Directory Structure

```
ai-predictive-maintenance/
├── run_pipeline.py          # Main pipeline orchestrator
├── run_dashboard.py         # Dashboard launcher
├── config.py                # Configuration settings
├── loaders.py               # Data loading utilities
│
├── data/
│   ├── raw/                 # Original NASA C-MAPSS files
│   └── processed/           # Cleaned and feature-engineered data
│
├── src/
│   ├── pipeline/            # Data processing scripts
│   │   ├── download_data.py
│   │   ├── verify_data.py
│   │   ├── clean_data.py
│   │   ├── data_prep_features.py
│   │   └── fix_scaler.py
│   │
│   ├── ai/                  # Model training and inference
│   │   ├── train_baseline_models.py
│   │   ├── train_xgboost_model.py
│   │   ├── inference.py
│   │   └── generate_report.py
│   │
│   └── dashboard/           # Web dashboard
│       ├── app.py
│       ├── charts.py
│       ├── results.py
│       ├── risk.py
│       ├── state.py
│       └── validation.py
│
├── models/                  # Trained model artifacts
├── results/                 # Performance reports and visualizations
├── notebooks/               # Jupyter notebooks for exploration
└── docs/                    # Documentation
```

## Key Lessons Learned

### Scaler Consistency is Critical
**Problem:** Initial cleaning created 4 separate scalers → inconsistent normalization → impossible to deploy.

**Solution:** Created `fix_scaler.py` to:
1. Fit ONE scaler on training data only
2. Transform all data with same scaler
3. Save scaler for inference

**Takeaway:** Preprocessing pipeline is as important as the model. Both must be versioned together.

### Feature Importance Analysis Reveals Deployment Issues
**Discovery:** `cycle_normalized` dominates (71%) but requires knowing when equipment fails.

**Lesson:** High accuracy doesn't guarantee production readiness. Always analyze what features actually mean and whether they'll be available at inference time.

## Project Status

### Completed ✅
- Data acquisition & validation (160K records)
- Data cleaning & normalization (0% missing values)
- Feature engineering (152 features created)
- Scaler correction (single consistent scaler)
- Model training & selection (XGBoost: 98% recall)
- Inference pipeline (production-ready)
- Performance reporting (4 visualizations)
- Dashboard deployment

### Next Steps ⏳
- Production deployment (without cycle_normalized)
- Flask API development (REST endpoints)
- 3D Unity Simulation with interactive scenarios
- Multi-Agent system (Anomaly Agent, Root Cause Agent, etc)

---

**Last Updated:** December 29, 2025 

**Status:** Phase 1 Complete ✅ | Dashboard Deployed ✅