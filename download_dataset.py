"""
Download Face Mask Detection dataset from Kaggle
"""
import os
import zipfile
import sys

def download_dataset():
    """Download dataset using Kaggle API"""
    try:
        # Check if kaggle is installed
        import kaggle
        
        # Set dataset details
        dataset = "andrewmvd/face-mask-detection"
        download_path = "data/raw"
        
        # Create directory if it doesn't exist
        os.makedirs(download_path, exist_ok=True)
        
        print(f"Downloading dataset: {dataset}")
        print(f"Destination: {download_path}")
        
        # Download dataset
        kaggle.api.dataset_download_files(dataset, path=download_path, unzip=True)
        
        print("✅ Dataset downloaded successfully!")
        
        # List downloaded files
        print("\nDownloaded files:")
        for file in os.listdir(download_path):
            print(f"  - {file}")
            
    except ImportError:
        print("❌ Kaggle library not found. Please install: pip install kaggle")
        print("\nAlternatively, download manually:")
        print("1. Visit: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection")
        print("2. Download the dataset")
        print("3. Extract to data/raw/")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nMake sure you have:")
        print("1. Kaggle account")
        print("2. API token (kaggle.json) in ~/.kaggle/ or %USERPROFILE%\\.kaggle\\")
        print("\nOr download manually from:")
        print("https://www.kaggle.com/datasets/andrewmvd/face-mask-detection")
        sys.exit(1)

if __name__ == "__main__":
    download_dataset()
