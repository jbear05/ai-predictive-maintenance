#!/usr/bin/env python3
"""
Download NASA C-MAPSS Dataset - Windows Version
Step 1.1 of CMMS AI Project
"""

import os
import urllib.request
import zipfile
import sys

def download_cmapss_dataset():
    """Download and extract NASA C-MAPSS dataset"""
    
    # Configuration
    url = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
    data_dir = "data\\raw"
    zip_path = os.path.join(data_dir, "cmapss.zip")
    
    # Create directory
    os.makedirs(data_dir, exist_ok=True)
    
    print("=" * 70)
    print("NASA C-MAPSS DATASET DOWNLOADER")
    print("=" * 70)
    print(f"\n📥 Starting download...")
    print(f"Source: {url}")
    print(f"Destination: {os.path.abspath(data_dir)}")
    print("-" * 70)
    
    try:
        # Download with progress bar
        def progress_hook(count, block_size, total_size):
            percent = min(int(count * block_size * 100 / total_size), 100)
            bar_length = 40
            filled = int(bar_length * percent / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\r[{bar}] {percent}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, zip_path, progress_hook)
        print("\n✅ Download complete!")
        
        # Get file size
        file_size = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"📦 File size: {file_size:.2f} MB")
        
        # Extract
        print("\n📂 Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            files = zip_ref.namelist()
            for i, file in enumerate(files, 1):
                zip_ref.extract(file, data_dir)
                print(f"\r  Extracting: {i}/{len(files)} files", end='', flush=True)
        
        print("\n✅ Extraction complete!")
        
        # List extracted files
        print("\n📁 Downloaded dataset files:")
        print("-" * 70)
        txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        for file in sorted(txt_files):
            filepath = os.path.join(data_dir, file)
            size = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✓ {file:<30} ({size:>6.2f} MB)")
        
        # Clean up zip file
        os.remove(zip_path)
        print(f"\n🗑️  Removed temporary zip file")
        
        print("\n" + "=" * 70)
        print("🎉 SUCCESS! Dataset ready for analysis!")
        print("=" * 70)
        print(f"\n📍 Dataset location: {os.path.abspath(data_dir)}")
        print(f"📊 Total files: {len(txt_files)}")
        print("\n✅ STEP 1.1 COMPLETE - Dataset acquired!")
        print("\nNext: Run 'python verify_data.py' to verify the data")
        
        return True
        
    except urllib.error.URLError as e:
        print(f"\n\n❌ Network Error: {e}")
        print("Check your internet connection and try again.")
        return False
    except zipfile.BadZipFile:
        print(f"\n\n❌ Error: Downloaded file is corrupted")
        print("Try running the script again.")
        return False
    except Exception as e:
        print(f"\n\n❌ Unexpected Error: {e}")
        return False

def main():
    """Main entry point"""
    print("\n")
    success = download_cmapss_dataset()
    
    if not success:
        print("\n⚠️  Download failed. Please try again or download manually from:")
        print("https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/")
        sys.exit(1)
    
    print("\n")

if __name__ == "__main__":
    main()