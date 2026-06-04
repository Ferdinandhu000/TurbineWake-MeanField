"""
preprocess_case5_lowTi_0_30.py  —  Preprocess and split case5_lowTi_0-30 dataset
================================================================================

Reads:
  - data/case5_lowTi_UWV_10min_bin1.nc (2500 frames)
  - data/case5_lowTi_UWV_10min_bin2.nc (2500 frames)
  - data/case5_lowTi_UWV_10min_bin3.nc (2500 frames)

Splits sequentially (8:1:1):
  - Train: first 6000 frames (all of bin1, all of bin2, first 1000 of bin3)
  - Val: next 750 frames (bin3 frames 1000 to 1749)
  - Test: last 750 frames (bin3 frames 1750 to 2499)

Output:
  - data/case5_lowTi_0-30/train/train_data.npy (6000, 120, 256, 3)
  - data/case5_lowTi_0-30/val/val_data.npy     (750, 120, 256, 3)
  - data/case5_lowTi_0-30/test/test_data.npy   (750, 120, 256, 3)
"""

import os
import shutil
import gc
import numpy as np

# ── configuration ─────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR     = os.path.join(DATA_DIR, 'case5_lowTi_0-30')

NX_ORIGINAL = 286
NY_ORIGINAL = 121
X_START_IDX = 30      # keeps x-indices 30..285, giving 256 x-points
FIELDS      = ['U', 'V', 'W']

def main():
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray is required. Install it with: pip install xarray netCDF4")

    # Clean and recreate output directories
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(OUT_DIR, split)
        if os.path.exists(split_dir):
            print(f"Removing old directory: {split_dir}")
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)

    bin1_path = os.path.join(DATA_DIR, 'case5_lowTi_UWV_10min_bin1.nc')
    bin2_path = os.path.join(DATA_DIR, 'case5_lowTi_UWV_10min_bin2.nc')
    bin3_path = os.path.join(DATA_DIR, 'case5_lowTi_UWV_10min_bin3.nc')

    for path in [bin1_path, bin2_path, bin3_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required data file not found: {path}")

    # Helper to load sliced and transposed data
    def load_slice(ds, start_t, end_t):
        arrays = []
        for field in FIELDS:
            # Crop y-dim to 120, x-dim from index 30 onwards
            arr = ds[field].values[start_t:end_t, X_START_IDX:, :120]  # (T, 256, 120)
            arr = arr.transpose(0, 2, 1)                              # (T, 120, 256)
            arrays.append(arr)
        return np.stack(arrays, axis=-1).astype(np.float32)            # (T, 120, 256, 3)

    # 1. TRAIN SPLIT: All of bin1 (2500), all of bin2 (2500), first 1000 of bin3 (1000)
    print("\n[1/3] Preparing TRAIN split (6000 frames) ...")
    
    print("Loading bin1...")
    with xr.open_dataset(bin1_path) as ds1:
        data_bin1 = load_slice(ds1, 0, 2500)
    print(f"Loaded bin1: {data_bin1.shape}")

    print("Loading bin2...")
    with xr.open_dataset(bin2_path) as ds2:
        data_bin2 = load_slice(ds2, 0, 2500)
    print(f"Loaded bin2: {data_bin2.shape}")

    print("Loading first 1000 frames of bin3...")
    with xr.open_dataset(bin3_path) as ds3:
        data_bin3_train = load_slice(ds3, 0, 1000)
    print(f"Loaded bin3 (train portion): {data_bin3_train.shape}")

    print("Concatenating train data...")
    train_data = np.concatenate([data_bin1, data_bin2, data_bin3_train], axis=0)
    print(f"Combined train data shape: {train_data.shape}")

    train_out_path = os.path.join(OUT_DIR, 'train', 'train_data.npy')
    print(f"Saving train data to {train_out_path} ...")
    np.save(train_out_path, train_data)
    print("Train split saved successfully.")

    # Free memory
    del data_bin1, data_bin2, data_bin3_train, train_data
    gc.collect()

    # 2. VAL SPLIT: bin3 frames 1000..1749 (750 frames)
    print("\n[2/3] Preparing VAL split (750 frames) ...")
    print("Loading bin3 frames 1000 to 1750...")
    with xr.open_dataset(bin3_path) as ds3:
        val_data = load_slice(ds3, 1000, 1750)
    print(f"Val data shape: {val_data.shape}")

    val_out_path = os.path.join(OUT_DIR, 'val', 'val_data.npy')
    print(f"Saving val data to {val_out_path} ...")
    np.save(val_out_path, val_data)
    print("Val split saved successfully.")

    # Free memory
    del val_data
    gc.collect()

    # 3. TEST SPLIT: bin3 frames 1750..2499 (750 frames)
    print("\n[3/3] Preparing TEST split (750 frames) ...")
    print("Loading bin3 frames 1750 to 2500...")
    with xr.open_dataset(bin3_path) as ds3:
        test_data = load_slice(ds3, 1750, 2500)
    print(f"Test data shape: {test_data.shape}")

    test_out_path = os.path.join(OUT_DIR, 'test', 'test_data.npy')
    print(f"Saving test data to {test_out_path} ...")
    np.save(test_out_path, test_data)
    print("Test split saved successfully.")

    # Final stats verification
    print("\n=========================================")
    print("Verification and Statistics:")
    print("=========================================")
    for split in ['train', 'val', 'test']:
        path = os.path.join(OUT_DIR, split, f"{split}_data.npy")
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            data_loaded = np.load(path, mmap_mode='r')
            print(f"Split '{split}': shape={data_loaded.shape}, file_size={size_mb:.2f} MB")
            # print stats per channel
            for c_idx, field in enumerate(FIELDS):
                ch = data_loaded[..., c_idx]
                print(f"  Channel {field} (min/max/mean): {ch.min():.4f} / {ch.max():.4f} / {ch.mean():.4f}")
    print("\nPreprocessing complete.")

if __name__ == '__main__':
    main()
