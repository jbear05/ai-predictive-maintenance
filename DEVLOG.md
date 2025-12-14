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

### Decisions
- 

### Problems/Solutions
-  Long file path when installing dependencies to venv
      Solution: Allow long file paths using admin terminal
-  Did not know how to add only the .gitignore so venv wouldn't commit
      Solution: Used bash terminal to go git add .gitignore
    

### Tomorrow's Plan
