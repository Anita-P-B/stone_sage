import os
import pandas as pd
from stone_sage.datasets.import_dataset import download_concrete_dataset # adjust if path is different
import hashlib

def compute_sha256(file_path):
    """Compute the SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)
        return sha256.hexdigest()
    except FileNotFoundError:
        return None

def load_or_download_data(path, force_download=False, expected_checksum=None,
                          debug= False):
    """
    Load the dataset if it exists and is valid. Otherwise, download it.

    Parameters:
        path (str): Path to the dataset CSV
        force_download (bool): If True, re-download regardless of presence
        expected_checksum (str or None): If given, verifies the file's integrity

    Returns:
        pd.DataFrame
    """
    #path = path[0] if isinstance(path, tuple) else path
    download_needed = force_download or not os.path.exists(path)

    if not download_needed and expected_checksum:
        actual_checksum = compute_sha256(path)
        if actual_checksum != expected_checksum:
            print("⚠️ Checksum mismatch! Re-downloading...")
            download_needed = True

    if download_needed:
        print(f"🔽 Downloading dataset to {path}...")
        try:
            download_concrete_dataset(data_path=path, debug = debug)
        except Exception as e:
            raise RuntimeError(f"Dataset download failed: {e}")

        if expected_checksum:
            actual_checksum = compute_sha256(path)
            #expected_checksum = expected_checksum[0] if isinstance(expected_checksum, tuple) else expected_checksum
            if actual_checksum != expected_checksum:
                raise ValueError("Downloaded file failed checksum validation.")

    else:
        print(f"📂 Dataset already exists at {path}. Loading...")

    return pd.read_csv(path)
