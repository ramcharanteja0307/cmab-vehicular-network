"""
Phase 1: Data Preparation
=========================
Loads raw datasets, removes dangerous columns, applies log transform,
validates everything, and saves cleaned datasets.

Usage:
    python src/data_preparation.py
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_raw_data():
    """Load all 4 raw CSV files and return as a dictionary of DataFrames."""
    datasets = {}
    for region in config.REGIONS:
        path = config.RAW_FILES[region]
        df = pd.read_csv(path)
        datasets[region] = df
        print(f"  Loaded {region}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return datasets


def drop_dangerous_columns(datasets):
    """
    Remove columns that cause data leakage or carry no useful information.
    Each column is dropped only if it exists in that dataset.
    """
    cleaned = {}
    for region, df in datasets.items():
        cols_to_drop = [col for col in config.DROP_COLUMNS if col in df.columns]
        cols_kept = [col for col in df.columns if col not in config.DROP_COLUMNS]
        
        df_clean = df.drop(columns=cols_to_drop)
        cleaned[region] = df_clean
        
        print(f"  {region}: dropped {len(cols_to_drop)} columns → {df_clean.shape[1]} remaining")
        if cols_to_drop:
            for col in cols_to_drop:
                print(f"    - {col}: {config.DROP_COLUMNS[col]}")
    
    return cleaned


def apply_log_transform(datasets):
    """
    Apply log transformation to datarate.
    
    Why: Raw datarate ranges from 851 to 271,000,000 bps.
    The distribution is heavily right-skewed (skewness ~2.5).
    After log(), skewness drops to ~0.0 — much better for modeling.
    
    log_datarate = ln(datarate)
    """
    for region, df in datasets.items():
        # Verify no zeros or negatives (log would fail)
        assert (df[config.TARGET_RAW] > 0).all(), \
            f"{region}: found zero or negative datarate values!"
        
        df[config.TARGET_LOG] = np.log(df[config.TARGET_RAW])
        
        raw_range = f"[{df[config.TARGET_RAW].min():,.0f} → {df[config.TARGET_RAW].max():,.0f}]"
        log_range = f"[{df[config.TARGET_LOG].min():.2f} → {df[config.TARGET_LOG].max():.2f}]"
        print(f"  {region}: datarate {raw_range} → log {log_range}")

    print("\n  Applying global MinMax normalization to log_datarate...")
    all_logs = np.concatenate([df[config.TARGET_LOG].values for df in datasets.values()])
    scaler = MinMaxScaler()
    scaler.fit(all_logs.reshape(-1, 1))
    
    for region, df in datasets.items():
        df[config.TARGET_NORM] = scaler.transform(df[config.TARGET_LOG].values.reshape(-1, 1)).flatten()
        norm_range = f"[{df[config.TARGET_NORM].min():.4f} → {df[config.TARGET_NORM].max():.4f}]"
        print(f"  {region}: norm {norm_range}")
    
    return datasets


def validate_common_features(datasets):
    """
    Verify that all 21 common features exist in every dataset.
    These features will be the input to the Neural Bandit.
    """
    print(f"  Checking {len(config.COMMON_FEATURES)} common features across all regions...")
    
    all_ok = True
    for region, df in datasets.items():
        missing = [f for f in config.COMMON_FEATURES if f not in df.columns]
        if missing:
            print(f"  ❌ {region} is missing: {missing}")
            all_ok = False
    
    if all_ok:
        print(f"  ✅ All {len(config.COMMON_FEATURES)} common features present in all 4 datasets")
    
    return all_ok


def validate_data_quality(datasets):
    """Run final quality checks on cleaned data."""
    all_ok = True
    
    for region, df in datasets.items():
        # Check for nulls
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            null_cols = df.columns[df.isnull().any()].tolist()
            print(f"  ❌ {region}: {null_count} null values in {null_cols}")
            all_ok = False
        
        # Check for infinities in log_datarate
        inf_count = np.isinf(df[config.TARGET_LOG]).sum()
        if inf_count > 0:
            print(f"  ❌ {region}: {inf_count} infinite values in log_datarate")
            all_ok = False
        
        # Check that no dropped columns snuck through
        leaked = [col for col in config.DROP_COLUMNS if col in df.columns]
        if leaked:
            print(f"  ❌ {region}: leaked columns still present: {leaked}")
            all_ok = False
    
    if all_ok:
        print("  ✅ All quality checks passed")
    
    return all_ok


def save_cleaned_data(datasets):
    """Save cleaned datasets to data/cleaned/."""
    os.makedirs(config.CLEANED_DATA_DIR, exist_ok=True)
    
    for region, df in datasets.items():
        path = config.CLEANED_FILES[region]
        df.to_csv(path, index=False)
        print(f"  Saved {region}: {df.shape[0]:,} rows x {df.shape[1]} columns → {os.path.basename(path)}")


def print_summary(datasets):
    """Print final summary of cleaned datasets."""
    print("\n" + "=" * 60)
    print("CLEANED DATASET SUMMARY")
    print("=" * 60)
    
    total_rows = 0
    for region, df in datasets.items():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in [config.TARGET_RAW, config.TARGET_LOG, config.TARGET_NORM]]
        total_rows += len(df)
        
        print(f"\n  {region.upper()}")
        print(f"    Rows:           {len(df):,}")
        print(f"    Total columns:  {df.shape[1]}")
        print(f"    Numeric features: {len(feature_cols)}")
        print(f"    Target (raw):   mean={df[config.TARGET_RAW].mean():,.0f} bps")
        print(f"    Target (log):   mean={df[config.TARGET_LOG].mean():.4f}, std={df[config.TARGET_LOG].std():.4f}")
        print(f"    Target (norm):  mean={df[config.TARGET_NORM].mean():.4f}, std={df[config.TARGET_NORM].std():.4f}")
    
    print(f"\n  TOTAL ROWS: {total_rows:,}")
    print(f"  COMMON FEATURES FOR NEURAL BANDIT: {len(config.COMMON_FEATURES)}")
    print("=" * 60)


def main():
    print("\n" + "=" * 60)
    print("PHASE 1: DATA PREPARATION")
    print("=" * 60)
    
    # Step 1: Load raw data
    print("\n[Step 1] Loading raw datasets...")
    datasets = load_raw_data()
    
    # Step 2: Drop dangerous columns
    print("\n[Step 2] Removing leakage / metadata columns...")
    datasets = drop_dangerous_columns(datasets)
    
    # Step 3: Apply log transformation
    print("\n[Step 3] Applying log transformation to datarate...")
    datasets = apply_log_transform(datasets)
    
    # Step 4: Validate common features
    print("\n[Step 4] Validating common feature set...")
    features_ok = validate_common_features(datasets)
    
    # Step 5: Quality checks
    print("\n[Step 5] Running quality checks...")
    quality_ok = validate_data_quality(datasets)
    
    if not features_ok or not quality_ok:
        print("\n❌ VALIDATION FAILED — fix issues above before proceeding.")
        sys.exit(1)
    
    # Step 6: Save cleaned data
    print("\n[Step 6] Saving cleaned datasets...")
    save_cleaned_data(datasets)
    
    # Summary
    print_summary(datasets)
    
    print("\n✅ Phase 1 complete. Cleaned data saved to data/cleaned/")
    print("   Next: Phase 2 — Prove Regional Distinctness")


if __name__ == "__main__":
    main()
