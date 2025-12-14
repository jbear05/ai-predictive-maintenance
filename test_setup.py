#!/usr/bin/env python3
"""
Test Python Environment Setup - Windows Version
Verify all required packages are installed
"""

import sys

def test_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        return True, "✅"
    else:
        return False, "❌"

def test_package(package_name, import_name=None):
    """Test if a package can be imported"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None

def main():
    """Main test function"""
    
    print("=" * 70)
    print("PYTHON ENVIRONMENT TEST - WINDOWS")
    print("=" * 70)
    
    # Test Python version
    print("\n📌 Python Version:")
    print("-" * 70)
    success, status = test_python_version()
    print(f"{status} Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    if not success:
        print("\n⚠️  WARNING: Python 3.8+ recommended. Please upgrade.")
    
    # Test packages
    print("\n📦 Required Packages:")
    print("-" * 70)
    
    packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('scikit-learn', 'sklearn'),
        ('xgboost', 'xgboost'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('plotly', 'plotly'),
        ('flask', 'flask'),
        ('streamlit', 'streamlit'),
        ('requests', 'requests'),
        ('jupyter', 'jupyter'),
    ]
    
    all_success = True
    installed = []
    missing = []
    
    for display_name, import_name in packages:
        success, version = test_package(display_name, import_name)
        if success:
            status = "✅"
            version_str = f"v{version}" if version != 'unknown' else ""
            print(f"{status} {display_name:<20} {version_str}")
            installed.append(display_name)
        else:
            status = "❌"
            print(f"{status} {display_name:<20} NOT INSTALLED")
            missing.append(display_name)
            all_success = False
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Installed: {len(installed)}/{len(packages)} packages")
    print(f"  Missing:   {len(missing)}/{len(packages)} packages")
    
    if all_success:
        print("\n🎉 SUCCESS! All required packages are installed!")
        print("\n✅ Your environment is ready for the CMMS AI project!")
        print("\nNext steps:")
        print("  1. Run: python download_data.py")
        print("  2. Run: python verify_data.py")
        print("  3. Start building your AI model!")
    else:
        print(f"\n⚠️  MISSING PACKAGES: {', '.join(missing)}")
        print("\nTo install missing packages, run:")
        print(f"  pip install {' '.join(missing)}")
        print("\nOr install all at once:")
        print("  pip install pandas numpy scipy scikit-learn xgboost matplotlib seaborn plotly flask streamlit requests jupyter")
    
    print("\n" + "=" * 70)
    
    return all_success

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)