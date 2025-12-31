#!/usr/bin/env python3
"""
Master Pipeline Script - AI Predictive Maintenance
==================================================
This script orchestrates the entire data pipeline from setup to dashboard.

Steps:
    1. Test Setup - Verify Python environment and required packages
    2. Download Data - Download NASA C-MAPSS dataset
    3. Verify Data - Validate the downloaded dataset
    4. Clean Data - Handle missing values, outliers, normalize
    5. Feature Prep - Engineer features and create train/val split
    6. Fix Scaler - Fit scaler on training data and transform validation
    7. Run Dashboard - Launch the Streamlit prediction dashboard

Usage:
    python run_pipeline.py              # Run full pipeline
    python run_pipeline.py --skip-download   # Skip download if data exists
    python run_pipeline.py --dashboard-only  # Only launch dashboard
"""


import subprocess
import sys
import os
from pathlib import Path
import argparse
from src.pipeline.terminal_colors import print_error, print_success, print_warning, print_header, Colors


def run_script(script_path: Path, description: str) -> bool:
    """
    Run a Python script and return success status.
    
    Parameters
    ----------
    script_path : Path
        Path to the Python script to run.
    description : str
        Description of what the script does.
    
    Returns
    -------
    bool
        True if script executed successfully, False otherwise.
    """
    print(f"Running: {script_path.name}")
    print(f"Purpose: {description}")
    print("-" * 50)
    
    try:
        # Determine the working directory based on script location
        # For scripts in src/pipeline, go up two levels to project root
        if script_path.parent.name == 'pipeline':
            cwd = script_path.parent.parent.parent  # src/pipeline -> src -> project_root
        elif script_path.parent.name == 'src':
            cwd = script_path.parent.parent  # src -> project_root
        else:
            cwd = script_path.parent
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=cwd,
            check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print_error(f"Script failed with exit code: {e.returncode}")
        return False
    except Exception as e:
        print_error(f"Error running script: {e}")
        return False


def check_data_exists(data_dir: Path) -> bool:
    """Check if raw data files already exist."""
    raw_data = data_dir / 'raw'
    if not raw_data.exists():
        return False
    
    # Check for at least one training file
    train_files = list(raw_data.glob('train_FD*.txt'))
    return len(train_files) >= 1


def check_processed_data_exists(data_dir: Path) -> bool:
    """Check if processed data files already exist."""
    processed_data = data_dir / 'processed'
    train_file = processed_data / 'train_processed.csv'
    val_file = processed_data / 'val_processed.csv'
    return train_file.exists() and val_file.exists()


def main():
    """Main pipeline orchestrator."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='AI Predictive Maintenance Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pipeline.py                  # Run full pipeline
    python run_pipeline.py --skip-download  # Skip download if data exists
    python run_pipeline.py --dashboard-only # Only launch dashboard
    python run_pipeline.py --no-dashboard   # Run pipeline without dashboard
        """
    )
    parser.add_argument(
        '--skip-download', 
        action='store_true',
        help='Skip data download if raw data already exists'
    )
    parser.add_argument(
        '--dashboard-only', 
        action='store_true',
        help='Only launch the dashboard (skip all processing)'
    )
    parser.add_argument(
        '--no-dashboard', 
        action='store_true',
        help='Run pipeline without launching dashboard at the end'
    )
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force re-run all steps even if data exists'
    )
    
    args = parser.parse_args()
    
    # Get project root directory
    project_root = Path(__file__).parent.resolve()
    src_dir = project_root / 'src'
    data_dir = project_root / 'data'
    
    # Print banner
    print(f"\n{'='*70}")
    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("    🔧 AI PREDICTIVE MAINTENANCE PIPELINE 🔧")
    print(f"{Colors.END}")
    print(f"{'='*70}\n")
    print(f"Project Root: {project_root}")
    print(f"Python: {sys.executable}")
    
    # Dashboard only mode
    if args.dashboard_only:
        print_header("LAUNCHING DASHBOARD")
        dashboard_script = project_root / 'run_dashboard.py'
        run_script(dashboard_script, "Launch Streamlit prediction dashboard")
        return
    
    steps_completed = 0
    total_steps = 6 if args.no_dashboard else 7
    
    # =========================================================================
    # STEP 1: Test Setup
    # =========================================================================
    print_header("TEST SETUP")
    setup_script = src_dir / 'pipeline' / 'test_setup.py'
    
    if setup_script.exists():
        if run_script(setup_script, "Verify Python environment and packages"):
            print_success("Setup verification complete!")
            steps_completed += 1
        else:
            print_error("Setup verification failed!")
            print("Please install required packages: pip install -r requirements.txt")
            response = input("\nContinue anyway? (y/n): ").strip().lower()
            if response != 'y':
                sys.exit(1)
    else:
        print_warning(f"Setup script not found: {setup_script}")
    
    # =========================================================================
    # STEP 2: Download Data
    # =========================================================================
    print_header("DOWNLOAD DATA")
    download_script = src_dir / 'pipeline' / 'download_data.py'
    
    if args.skip_download and check_data_exists(data_dir) and not args.force:
        print_success("Raw data already exists - skipping download")
        print(f"   Location: {data_dir / 'raw'}")
        steps_completed += 1
    elif download_script.exists():
        if run_script(download_script, "Download NASA C-MAPSS dataset"):
            print_success("Data download complete!")
            steps_completed += 1
        else:
            print_error("Data download failed!")
            sys.exit(1)
    else:
        print_warning(f"Download script not found: {download_script}")
    
    # =========================================================================
    # STEP 3: Verify Data
    # =========================================================================
    print_header("VERIFY DATA")
    verify_script = src_dir / 'pipeline' / 'verify_data.py'
    
    if verify_script.exists():
        if run_script(verify_script, "Validate downloaded dataset"):
            print_success("Data verification complete!")
            steps_completed += 1
        else:
            print_error("Data verification failed!")
            sys.exit(1)
    else:
        print_warning(f"Verify script not found: {verify_script}")
    
    # =========================================================================
    # STEP 4: Clean Data
    # =========================================================================
    print_header("CLEAN DATA")
    clean_script = src_dir / 'pipeline' / 'clean_data.py'
    
    if clean_script.exists():
        if run_script(clean_script, "Handle missing values, outliers, and normalize"):
            print_success("Data cleaning complete!")
            steps_completed += 1
        else:
            print_error("Data cleaning failed!")
            sys.exit(1)
    else:
        print_warning(f"Clean script not found: {clean_script}")
    
    # =========================================================================
    # STEP 5: Feature Preparation
    # =========================================================================
    print_header("FEATURE PREPARATION")
    feature_script = src_dir / 'pipeline' / 'data_prep_features.py'
    
    if feature_script.exists():
        if run_script(feature_script, "Engineer features and create train/val split"):
            print_success("Feature preparation complete!")
            steps_completed += 1
        else:
            print_error("Feature preparation failed!")
            sys.exit(1)
    else:
        print_warning(f"Feature script not found: {feature_script}")
    
    # =========================================================================
    # STEP 6: Fix Scaler
    # =========================================================================
    print_header("FIX SCALER")
    scaler_script = src_dir / 'pipeline' / 'fix_scaler.py'
    
    if scaler_script.exists():
        if run_script(scaler_script, "Fit scaler on training data and save"):
            print_success("Scaler fix complete!")
            steps_completed += 1
        else:
            print_error("Scaler fix failed!")
            sys.exit(1)
    else:
        print_warning(f"Scaler script not found: {scaler_script}")
    
    # =========================================================================
    # STEP 7: Run Dashboard (optional)
    # =========================================================================
    if not args.no_dashboard:
        print_header("LAUNCHING DASHBOARD")
        dashboard_script = project_root / 'run_dashboard.py'
        
        if dashboard_script.exists():
            print_success(f"Pipeline complete! {steps_completed}/{total_steps-1} steps succeeded.")
            print("\n🚀 Launching prediction dashboard...")
            print("   Press Ctrl+C to stop the dashboard\n")
            run_script(dashboard_script, "Launch Streamlit prediction dashboard")
        else:
            print_warning(f"Dashboard script not found: {dashboard_script}")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"{Colors.BOLD}{Colors.GREEN}")
    print("    ✅ PIPELINE EXECUTION COMPLETE!")
    print(f"{Colors.END}")
    print(f"{'='*70}")
    print(f"\nSteps Completed: {steps_completed}/{total_steps if args.no_dashboard else total_steps-1}")
    print(f"\nOutput Files:")
    print(f"   📁 Raw Data:       {data_dir / 'raw'}")
    print(f"   📁 Processed Data: {data_dir / 'processed'}")
    print(f"   📁 Models:         {project_root / 'models'}")
    
    if args.no_dashboard:
        print(f"\n💡 To launch the dashboard later, run:")
        print(f"   python run_dashboard.py")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
