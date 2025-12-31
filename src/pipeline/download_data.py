#!/usr/bin/env python3
"""
Download NASA C-MAPSS Dataset - Windows Version
Step 1.1 of CMMS AI Project
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import urllib.request
import zipfile
import typing as t # For type hinting complex structures
from config import config
from terminal_colors import Colors

# dataset URL, directory paths, and zip file path
URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
data_dir = config.paths.raw_data  # Uses Path object
zip_path = data_dir / "cmapss.zip"

def progress_hook(count: int, block_size: int, total_size: int) -> None:
    """
    Callback function used by urllib.request to display download progress.

    Parameters
    ----------
    count : int
        The number of blocks already transferred.
    block_size : int
        The size of each block in bytes.
    total_size : int
        The total size of the file in bytes.
    """
    # Calculate percentage transferred, capping at 100%
    percent: int = min(int(count * block_size * 100 / total_size), 100)
    
    # Render the progress bar visually
    bar_length: int = 40
    filled: int = int(bar_length * percent / 100)
    bar: str = '█' * filled + '░' * (bar_length - filled)
    
    # Print the progress bar to the console without a newline (using \r)
    print(f"\r[{bar}] {percent}%", end='', flush=True)


def download_cmapss_dataset() -> bool:
    """
    Handles the entire process of downloading, extracting, organizing, 
    and cleaning up the NASA C-MAPSS dataset files.

    Returns
    -------
    bool
        True if the download and extraction were successful, False otherwise.
    """
    
    # Configuration
    
    # Create the data directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)
    
    # --- Start Report ---
    print(f"{'='*70}")
    print(f"{Colors.BOLD}{Colors.BLUE}NASA C-MAPSS DATASET DOWNLOADER (Step 1.1){Colors.END}")
    print(f"{'='*70}")
    print(f"\n📥 Starting download...")
    print(f"Source: {URL}")
    print(f"Destination: {os.path.abspath(data_dir)}")
    print(f"{'-'*70}")
    
    try:
        # 1. Download File with Progress
        # The progress_hook defined above is used for visual feedback
        urllib.request.urlretrieve(URL, zip_path, progress_hook)
        print(f"\n{Colors.GREEN}✅ Download complete!{Colors.END}")
        
        # Get file size for confirmation
        file_size: float = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"📦 File size: {file_size:.2f} MB")
        
        # 2. Extract Files
        print(f"\n📂 Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            files: t.List[str] = zip_ref.namelist()
            
            # Extract each file and provide a simple counter
            for i, file in enumerate(files, 1):
                zip_ref.extract(file, data_dir)
                print(f"\r  Extracting: {i}/{len(files)} files", end='', flush=True)
        
        print(f"\n{Colors.GREEN}✅ Extraction complete!{Colors.END}")
        
        # 3. Final Verification and Cleanup
        
        # List and confirm extracted files
        print(f"\n📁 Downloaded dataset files:")
        print(f"{'-'*70}")
        # Filter files for the text data files (.txt)
        txt_files: t.List[str] = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        
        for file in sorted(txt_files):
            filepath: str = os.path.join(data_dir, file)
            size: float = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✓ {file:<30} ({size:>6.2f} MB)")
        
        # Clean up the temporary zip file
        os.remove(zip_path)
        print(f"\n🗑️  Removed temporary zip file: {zip_path}")
        
        # --- Final Summary ---
        print(f"\n{'='*70}")
        print(f"{Colors.GREEN}🎉 SUCCESS! Dataset ready for analysis!{Colors.END}")
        print(f"{'='*70}")
        print(f"\n📍 Dataset location: {os.path.abspath(data_dir)}")
        print(f"📊 Total data files: {len(txt_files)}")
        print(f"\n✅ STEP 1.1 COMPLETE - Dataset acquired!")
        print(f"\nNext: Run 'python clean_data.py' to process the data")
        
        return True
        
    except urllib.error.URLError as e:
        # Handle network-related errors (e.g., no internet, bad URL)
        print(f"\n\n{Colors.RED}❌ Network Error: {e}{Colors.END}")
        print("Check your internet connection and verify the URL is correct.")
        return False
    except zipfile.BadZipFile:
        # Handle cases where the downloaded file is incomplete or corrupted
        print(f"\n\n{Colors.RED}❌ Error: Downloaded zip file is corrupted{Colors.END}")
        print("Try deleting the .zip file and running the script again.")
        return False
    except Exception as e:
        # Catch any other unexpected errors
        print(f"\n\n{Colors.RED}❌ Unexpected Error: {e}{Colors.END}")
        return False


def main() -> None:
    """
    Main entry point of the script. Calls the download function and exits 
    with a non-zero code if the download fails.
    """
    print("\n")
    success: bool = download_cmapss_dataset()
    
    if not success:
        print(f"\n{Colors.YELLOW}⚠️  Download failed. You may need to download the dataset manually from:{Colors.END}")
        print(URL)
        sys.exit(1)
    
    print("\n")

if __name__ == "__main__":
    main()