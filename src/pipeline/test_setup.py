#!/usr/bin/env python3
"""
Test Python Environment Setup - Windows Version
Verify all required packages are installed
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import types # Import types for robust checking
import importlib.metadata
from terminal_colors import Colors, print_error, print_success, print_warning, print_header

def test_python_version() -> tuple[bool, str]:
    """
    Checks the installed Python version against the recommended minimum (3.8+).

    Returns
    -------
    tuple[bool, str]
        A tuple containing (success status, status emoji).
    """
    version = sys.version_info
    
    # Print the full version number for user visibility
    print(f"Recommended: Python {version.major}.{version.minor}.{version.micro}")
    
    # Check if the major version is 3 and minor version is 8 or greater
    if version.major == 3 and version.minor >= 8:
        return True, "✅"
    else:
        return False, "❌"

def test_package(package_name: str, import_name: str | None = None) -> tuple[bool, str | None]:
    """
    Attempts to dynamically import a package to test for its presence.

    Parameters
    ----------
    package_name : str
        The common display name of the package (e.g., 'scikit-learn').
    import_name : str, optional
        The actual name used in the 'import' statement (e.g., 'sklearn'). 
        Defaults to package_name if not provided.

    Returns
    -------
    tuple[bool, str | None]
        A tuple containing (success status, package version string, or None if missing).
    """
    if import_name is None:
        import_name = package_name
    
    try:
        # Dynamically import the module using its import name
        module: types.ModuleType = __import__(import_name) 
        
        # Try to get version using importlib.metadata first (preferred for modern packages)
        try:
            version: str = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            # Fall back to module's __version__ attribute
            version: str = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        # If the import fails, the package is not installed
        return False, None

def main():
    """
    Main execution function. Runs all environment tests and prints a comprehensive 
    summary report indicating installed and missing packages.
    """
    
    # --- Report Header ---
    print_header("PYTHON ENVIRONMENT TEST")
    
    # --- Test 1: Python version ---
    print(f"\nPython Version Check:")
    print(f"{'-'*70}")
    success, status = test_python_version()
    print(f"{status} Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    if not success:
        print_warning("⚠️ Recommended Python version is 3.8 or higher.")
    
    # --- Test 2: Required Packages ---
    print_header("REQUIRED PACKAGES CHECK")
    
    # List of all required packages for Phases 1-3 of the project plan
    packages: list[tuple[str, str]] = [
        ('pandas', 'pandas'),           # Data Manipulation
        ('numpy', 'numpy'),             # Numerical Operations
        ('scipy', 'scipy'),             # Statistical Functions (e.g., zscore)
        ('scikit-learn', 'sklearn'),    # Core ML Algorithms (MinMaxScaler, Models)
        ('xgboost', 'xgboost'),         # Advanced ML Model
        ('matplotlib', 'matplotlib'),   # Visualization
        ('seaborn', 'seaborn'),         # Enhanced Visualization
        ('plotly', 'plotly'),           # Interactive Visualization (Optional but useful)
        ('flask', 'flask'),             # Backend API Framework
        ('requests', 'requests'),       # Testing API endpoints
        ('streamlit', 'streamlit'),     # Dashboard Framework
        ('jupyter', 'jupyter'),         # Interactive Development (Notebooks)
    ]
    
    all_success: bool = True
    installed: list[str] = []
    missing: list[str] = []
    
    # Iterate through the list and test each package
    for display_name, import_name in packages:
        success, version = test_package(display_name, import_name)
        if success:
            version_str = f"v{version}" if version != 'unknown' else ""
            print_success(f"{display_name:<20} {version_str}")
            installed.append(display_name)
        else:
            print_error(f"{display_name:<20} NOT INSTALLED")
            missing.append(display_name)
            all_success = False
    
    # --- Summary and Guidance ---
    print_header("SUMMARY")

    print(f"  Installed: {len(installed)}/{len(packages)} packages")
    print(f"  Missing:   {len(missing)}/{len(packages)} packages")
    
    if all_success:
        print_success("SUCCESS! All required packages are installed!")
        print_success("Your environment is ready for the CMMS AI project!")
    else:
        print_warning(f"⚠️  MISSING PACKAGES: {', '.join(missing)}")
        print("\nTo install missing packages, run:")
        print(f"  pip install {' '.join(missing)}")
        
    print(f"\n{'='*70}")
    
    return all_success

if __name__ == "__main__":
    # If the environment check fails, exit the script with a non-zero status code (1)
    success = main()
    if not success:
        sys.exit(1)