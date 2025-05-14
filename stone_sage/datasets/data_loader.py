import os
import pandas as pd
from stone_sage.datasets.import_dataset import download_concrete_dataset # adjust if path is different
import hashlib
from sklearn.model_selection import train_test_split


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

def split_and_save_partitions(df, run_dir, val_ratio=0.1, test_ratio=0.1, random_state=42):
    print("🔪 Splitting dataset into train/val/test partitions...")

    # First, split off the test set
    train_val_df, test_df = train_test_split(df, test_size=test_ratio, random_state=random_state)
    # Then split the remaining into train/val
    train_df, val_df = train_test_split(train_val_df, test_size=val_ratio / (1 - test_ratio), random_state=random_state)

    # Add partition column
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["partition"] = "train"
    val_df["partition"] = "val"
    test_df["partition"] = "test"

    # Combine them back
    full_df = pd.concat([train_df, val_df, test_df], axis=0).sort_index()

    # Ensure run_dir exists
    os.makedirs(run_dir, exist_ok=True)

    # Save with new column
    output_path = os.path.join(run_dir, "dataset_with_partitions.csv")
    full_df.to_csv(output_path, index=False)

    print(f"💾 Partitioned dataset saved to: {output_path}")
    validate_partition_leakage(train_df,val_df,test_df)
    return full_df, train_df, val_df, test_df

def validate_partition_leakage(train_df, val_df, test_df, id_columns=None):
    """
    Ensures that train/val/test partitions are disjoint.
    Assumes indices or ID columns are unique identifiers.

    Args:
        train_df, val_df, test_df (pd.DataFrame): partitioned dataframes
        id_columns (list or None): columns to use as unique identifiers. If None, uses index.
    """
    def get_ids(df):
        return df.index if id_columns is None else df[id_columns].apply(lambda row: "_".join(row.values.astype(str)), axis=1)

    train_ids = set(get_ids(train_df))
    val_ids = set(get_ids(val_df))
    test_ids = set(get_ids(test_df))

    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids & test_ids

    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(f"Partition leakage detected!\n"
                         f"Train/Val overlap: {len(overlap_tv)}\n"
                         f"Train/Test overlap: {len(overlap_tt)}\n"
                         f"Val/Test overlap: {len(overlap_vt)}")
    print("✅ No partition leakage detected.")


