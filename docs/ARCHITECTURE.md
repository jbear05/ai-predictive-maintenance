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

## Air-gapped deployment readiness
### ✅ What's Already Air-Gap Compatible
**1. Zero Cloud Dependencies**
- Local Inference: All ML inference runs locally using joblib.load() for models
- Localhost Hosting: Dashboard operates on localhost:8050 with no internet requirements
- No Telemetry: No external API calls, cloud storage, or telemetry

**2. Self-Contained Artifacts**
```
models/
├── xgboost_model.pkl       # Complete trained model (~50MB)
├── scaler.pkl              # Preprocessing state
└── scaler_columns.json     # Feature schema metadata
```

**3. Reproducible Environment**
- requirements.txt with pinned dependency versions
- Virtual environment for isolated deployment
- No dynamic package resolution at runtime

**4. Basic Security Controls**
- 100MB max uploads
- Row/column limits to prevent resource exhaustion
- 500MB max usage
- Robust CSV parsing


### ⚠️ Critical Gaps for DOE Deployment (6-8 weeks to resolve)

#### Phase 1: Security Essentials (3 weeks)

**Dependency Vendoring (CRITICAL):**

- Current: pip install requires internet access.
- Need: wheels/ directory with all packages for offline installation.
- Implementation: pip download -r requirements.txt -d ./wheels/

**Model Integrity Verification (CRITICAL):**

- Current: joblib.load() can execute arbitrary code during unpickling.
- Need: HMAC-SHA256 signatures + SHA256SUMS checksums.
- Risk: Malicious model replacement could execute code.

**User Authentication & RBAC (CRITICAL):**

- Current: No authentication—anyone can access dashboard.
- Need: Role-based access control (Viewer/Engineer/Admin roles).
- Integration: Link to site LDAP/Active Directory.

**Comprehensive Audit Logging (CRITICAL):**

- Current: Minimal logging.
- Need: Tamper-evident logs capturing WHO, WHAT, WHEN for every operation.
- Format: JSON logs with file hashes, usernames, timestamps, predictions.

#### Phase 2: Data Security (2 weeks)

**Enhanced Input Validation:**

- Add file signature verification (magic bytes).
- Implement CSV sandboxing with bounds checking.
- Validate sensor ranges against physical limits (e.g., temperature 400-700°R).

**Network Isolation Verification:**

- Add startup check to verify air-gap (fail if network detected).
- Test against known external IPs (8.8.8.8, pypi.org, github.com).

#### Phase 3: Operations (2 weeks)

**Immutable Model Deployment:**

- Make models read-only: chmod 444 models/*.pkl
- Use chattr +i (Linux) or icacls /deny (Windows).
- Prevent modification even by administrators.

**Installation Documentation:**

- Create INSTALL_AIRGAP.md with offline transfer procedures.
- Document "Two-Person Integrity" for media transfer.
- Include GPG signature verification steps.

#### Phase 4: Compliance (1 week)

**Security Documentation:**

- System Security Plan (SSP).
- NIST 800-171 controls mapping.
- Threat model with mitigations.
- Incident response procedures.

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
- Air-gapped deployment preparation
- Flask API development (REST endpoints)
- 3D Unity Simulation with interactive scenarios
- Multi-Agent system (Anomaly Agent, Root Cause Agent, etc)

---

**Last Updated:** December 29, 2025 

**Status:** Phase 1 Complete ✅ | Dashboard Deployed ✅