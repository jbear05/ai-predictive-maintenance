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
- Create train/test split (80/20), engineer 12+ features (rolling averages, rate of change, deviation from baseline), and generate binary target variable ("Will fail in next 48 cycles?")

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

### What I Did
**Afternoon (1pm-5pm):**
- Reviewed my code for data prep and dived into how pandas and numpy work for a further understanding on what I am doing with my python scripts.

**Evening (7pm-11pm):**
- Continued to review my code and furthered my understanding of pandas and numpy

### Decisions
- Pivoted my working style to learn the technologies I am using better

### Problems/Solutions
-  None

### Tomorrow's Plan
- Learn more about training AI models
- Train XGBoost model with hyperparameter tuning using GridSearchCV across 20+ parameter combinations
- Set up Python environment with required libraries and train baseline Logistic Regression + Random Forest models
- Change data_prep_features so results are sent to new results folder


## Day 5

### Goals for Today
- Learn more about training AI models
- Train XGBoost model with hyperparameter tuning using GridSearchCV across 20+ parameter combinations
- Set up Python environment with required libraries and train baseline Logistic Regression + Random Forest models
- Change data_prep_features so results are sent to new results folder

### What I Did
**Afternoon (2pm-5pm):**
- Sci-kit learn practice
- Created new notebook for ai analysis

**Afternoon-Evening (5pm-12am):**
- train_baseline_models.py created and model_comparison.txt generated
- baseline models are trained with random forest being the baseline to beat
- Train XGBoost model with hyperparameter tuning using GridSearchCV across 20+ parameter combinations

### Decisions
- Using random forest model as baseline to beat for XGBoost
- I have decided to keep my prediction window to 48 cycles (not hours) since changing it down to match my initial goal of 48 hour prediction is not possible due to no cycle to time conversion provided in the data.
      What cycles could mean: 1 Cycle = 1 complete flight
      The time for this could range from about 1 to 2 weeks in advance
      This is valuable for ordering parts, scheduling maintenance windoes, route planning around maintenance, etc.
- Decided to keep data_prep_features results being sent to processed data for now


### Problems/Solutions
- RUL column in training data was giving AI the answer thus providing false accuracy results
      Solution: Dropping the column when training gave realistic accuracy results
- Operational Cycles do not directly translate to hours
      Solution: Address the cycle-to-time conversion issue in my documentation

### Tomorrow's Plan
- Commit train_xgboost_model.py if overnight training worked
- Update documentation with new xgboost model information
- Select best performing model, create inference pipeline, save as .pkl file, and generate performance report with confusion matrix, ROC curve, and feature importance charts


## Day 6

### Goals for Today
- Commit train_xgboost_model.py if overnight training worked
- Update documentation with new xgboost model information

### What I Did
**Evening (7pm-11pm):**
- Ensured model was working as planned
- Committed new script and updated documentation

### Decisions
- None


### Problems/Solutions
- None

### Tomorrow's Plan
- Select best performing model, create inference pipeline, save as .pkl file, and generate performance report with confusion matrix, ROC curve, and feature importance charts


## Day 7

### Goals for Today
- Select best performing model, create inference pipeline, save as .pkl file, and generate performance report with confusion matrix, ROC curve, and feature importance charts

### What I Did
**Afternoon (12pm-5pm):**
- Fixed scaler issue and retrained model

**Evening (7pm-11pm):**
- Committed new changes and fix_scaler.py

### Decisions
- None


### Problems/Solutions
- When cleaning the datasets, I used different scalers for each raw data file, which makes it impossible to choose one of the four scalers when normalizing new data
      Solution: Rescale train_processed.csv and retrain my model

### Tomorrow's Plan
- Create inference pipeline and documentation


## Day 8

### Goals for Today
- Create inference pipeline and documentation

### What I Did
**Afternoon-Evening (12pm-11pm):**
- inference.py created
- generate_report.py created and report generated with 4 images
- Documentation updated
- File paths updated

### Decisions
- None


### Problems/Solutions
- None

### Tomorrow's Plan
- Code Review and refactor any code that can be improved


## Day 9

### Goals for Today
- Code Review and refactor any code that can be improved

### What I Did
**Evening (6pm-11pm):**
- Code Review

### Decisions
- None


### Problems/Solutions
- None

### Tomorrow's Plan
- Code Review and refactor any code that can be improved