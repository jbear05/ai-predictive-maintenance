# System Architecture

## High-Level Overview
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐
│  NASA C-MAPSS   │───▶│  Data Processing │───▶│  Processed  │
│  Raw Data (.txt)│    │  & Feature Eng   │    │  Data (.csv)│
└─────────────────┘    └──────────────────┘    └─────────────┘
```

## Components

### 1. Data Processing Layer

**Verification Script** (`verify_data.py`)
- Loads all C-MAPSS training files (train_FD001-004.txt)
- Validates dataset meets minimum 50,000 record requirement
- Provides statistical summary and data quality checks
- Status: ✅ Complete (Step 1.1)

**Cleaning Script** (`clean_data.py`)
- Handles missing values (dropna)
- Removes outliers using 3-sigma rule (Z-score > 3)
- Min-Max normalization (0-1 scale) for all sensor columns
- Skips constant features to prevent scaling errors
- Outputs: `train_FD001_cleaned.csv` through `train_FD004_cleaned.csv`
- Status: ✅ Complete (Step 1.2)

**Feature Engineering Script** (`data_prep_features.py`)
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
- Status: ✅ Complete (Step 1.3)

### 2. Model Training Layer
[Not yet implemented - Step 2.1-2.3 pending]

### 3. Inference Layer
[Not yet implemented - Step 3.1-3.3 pending]

### 4. Dashboard Layer
[Not yet implemented - Step 4.1-4.3 pending]

## Data Flow

**Current Data Pipeline:**
```
Raw Data (data/raw/)
    ├── train_FD001.txt (20,631 records)
    ├── train_FD002.txt (53,759 records)
    ├── train_FD003.txt (24,720 records)
    └── train_FD004.txt (61,249 records)
           ↓
    [verify_data.py]
           ↓
    Verification Report
           ↓
    [clean_data.py]
           ↓
Cleaned Data (data/processed/)
    ├── train_FD001_cleaned.csv (19,337 records)
    ├── train_FD002_cleaned.csv (53,759 records)
    ├── train_FD003_cleaned.csv (22,794 records)
    └── train_FD004_cleaned.csv (61,249 records)
           ↓
    [data_prep_features.py]
           ↓
Combined Dataset
    └── 157,139 total records, 260 engines
           ↓
    [Feature Engineering]
           ↓
    173 engineered features created
           ↓
    [Train/Val Split - 80/20]
           ↓
Final Processed Data (data/processed/)
    ├── train_processed.csv (126,954 records, 202 columns)
    ├── val_processed.csv (30,185 records, 202 columns)
    ├── feature_documentation.csv (173 features)
    └── data_quality_report.txt
```

## Security Considerations

**Current Implementation:**
- All data processing is local (no external API calls)
- No sensitive data transmission
- Files stored locally in project directory
- Uses standard Python libraries (pandas, numpy, scikit-learn, scipy)

**Privacy by Design:**
- Data never leaves local machine
- No cloud dependencies in data processing pipeline
- Suitable for air-gapped deployment preparation

## Performance

**Data Processing Performance:**
- Combined dataset: 157,139 records processed
- Feature engineering: 173 features created per record
- Final dataset size: 202 columns × 157,139 rows
- Memory usage: Manageable on standard development machine
- Processing time: Approximately 2-5 minutes for full pipeline (hardware dependent)

**Data Quality Metrics:**
- Missing values: 0.00% (meets <2% requirement)
- Outlier removal: ~1-3% of records removed per file
- All sensors normalized to 0-1 scale

## Future Architecture
- Add Flask API layer for CMMS integration
- Add SQLite database for historical predictions
- Multi-agent architecture for specialized tasks