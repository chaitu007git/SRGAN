import os
import zipfile

def download_and_extract_dataset(download_path, extract_path):
    """
    This function extracts a dataset from a zip file.

    Args:
        download_path (str): The path to the downloaded zip file.
        extract_path (str): The path where the dataset will be extracted.

    Returns:
        None
    """
    print("[INFO] Checking if the zip file exists...")
    if not os.path.exists(download_path):
        raise FileNotFoundError(f"The file {download_path} does not exist.")
    
    print("[INFO] Extracting the dataset...")
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"[INFO] Dataset extracted to {extract_path}")

if __name__ == "__main__":
    # Define the paths
    download_path = 'E:\SRGAN Project Zip\Data\DATA_1024.zip'  # Path to the zip file
    extract_path = '/content/'  # Path to extract the dataset
    
    # Ensure the extraction directory exists
    os.makedirs(extract_path, exist_ok=True)

    # Extract the dataset
    download_and_extract_dataset(download_path, extract_path)