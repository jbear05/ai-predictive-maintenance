# AI Predictive Maintenance - Development Log

## Day 1

### Goals for Today
- Initial Setup

### What I Did
**End of day (9pm-12am):**
- Downloaded NASA dataset
- Documentation setup
- Created project structure:
```
  ai-predictive-maintenance/
  ├── data/
  ├── notebooks/
  ├── src/
  └── docs/
```


## Day 2

### Goals for Today
-  Extract files to project directory
-  Verify dataset loads in Python/pandas
-  Confirm ≥100,000 records available (across all FD datasets)
-  Document file location and record count

### What I Did
**Afternoon (5pm-7pm):**
- venv with dependencies
- test_setup file to check dependencies
- download_data file to download CMAPPS data set automatically
- verify_data file to verify that the data set is ready for cleaning and preparation
- .gitignore for venv

**End of day (9pm-12am):**
- Added .gitignore to repository
- All changes staged and committed to repository

### Decisions
- Just followed the plan for today

### Problems/Solutions
-  Long file path when installing dependencies to venv
      Solution: Allow long file paths using admin terminal
-  Did not know how to add only the .gitignore so venv wouldn't commit
      Solution: Used bash terminal to go git add .gitignore
    

### Tomorrow's Plan
- Clean dataset by removing null values, handling outliers (>3 std dev), and normalizing sensor readings to 0-1 scale


## Day 3

### Goals for Today
- Clean dataset by removing null values, handling outliers (>3 std dev), and normalizing sensor readings to 0-1 scale
- Create train/test split (80/20), engineer 12+ features (rolling averages, rate of change, deviation from baseline), and generate binary target variable ("Will fail in next 48 hours?")

### What I Did
**Afternoon (12pm-3pm):**
- clean_data.py created
- In file documentation updated

**Afternoon-Evening (4pm-10pm):**
- clean_data and verify_data now go through all files instead of a specific one
- data_prep_features.py created to split data into an 80/20 split of train/validation data
- data quality report and feature documentation generated

### Decisions
- Following standard docstring format from now on when commenting code
- Train/Test split will be 80/20, splitting the sorted training data by row-count, thus ensuring that we train on the earlier history and validate on the later history.

### Problems/Solutions
-  clean_data.py -> The outlier removal deleted ALL rows!
      Solution: Use a more robust outlier removal method in clean_data.py remove_outliers_3sigma function
-  data_prep_features.py -> the checkmark character ✓ can't be encoded in Windows' default
      Solution: encoding='utf-8'

### Tomorrow's Plan
- Set up Python environment with required libraries and train baseline Logistic Regression + Random Forest models
- Train XGBoost model with hyperparameter tuning using GridSearchCV across 20+ parameter combinations


## Day 4

### Goals for Today
- Learn more about pandas and numpy
- Set up Python environment with required libraries and train baseline Logistic Regression + Random Forest models
- Train XGBoost model with hyperparameter tuning using GridSearchCV across 20+ parameter combinations

### What I Did
**Afternoon (1pm-5pm):**
- Reviewed my code for data prep and dived into how pandas and numpy work for a further understanding on what I am doing with my python scripts.

### Decisions
- 

### Problems/Solutions
-  

### Tomorrow's Plan
- 