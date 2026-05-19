"""
Phase 2: Prove Regional Distinctness
=====================================
Trains one LightGBM model per region using the 21 common features.
Tests each model on every region's data to produce a 4×4 RMSE matrix.

If the matrix shows that own-region RMSE is much lower than cross-region RMSE,
it proves that each region has unique signal behavior and a single model
cannot generalize across all regions.

Usage:
    python src/prove_regional_distinctness.py
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_cleaned_data():
    """Load all 4 cleaned datasets."""
    datasets = {}
    for region in config.REGIONS:
        path = config.CLEANED_FILES[region]
        df = pd.read_csv(path)
        datasets[region] = df
        print(f"  Loaded {region}: {df.shape[0]:,} rows")
    return datasets


def prepare_train_test_splits(datasets):
    """
    For each region, split into 80% train and 20% test.
    Uses ONLY the 21 common features — so all models are comparable.
    """
    splits = {}
    for region, df in datasets.items():
        X = df[config.COMMON_FEATURES]
        y = df[config.TARGET_NORM]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_SEED
        )

        splits[region] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        print(f"  {region}: train={len(X_train):,}, test={len(X_test):,}")

    return splits


def train_models(splits):
    """
    Train one LightGBM model per region using default hyperparameters.
    (We intentionally use defaults here — PSO optimization comes in Phase 3.)
    """
    models = {}
    for region in config.REGIONS:
        data = splits[region]

        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            random_state=config.RANDOM_SEED,
            verbose=-1,  # suppress training output
        )

        model.fit(data["X_train"], data["y_train"])
        models[region] = model

        # Quick check: own-region train RMSE
        train_pred = model.predict(data["X_train"])
        train_rmse = np.sqrt(mean_squared_error(data["y_train"], train_pred))
        print(f"  {region}: trained (train RMSE = {train_rmse:.4f})")

    return models


def build_cross_region_matrix(models, splits):
    """
    Test every model on every region's test set.
    Produces a 4×4 RMSE matrix.

    matrix[i][j] = RMSE when model trained on region i predicts region j's test data.

    Diagonal = own-region (should be LOW).
    Off-diagonal = cross-region (should be HIGH if regions are truly different).
    """
    regions = config.REGIONS
    matrix = np.zeros((len(regions), len(regions)))

    for i, train_region in enumerate(regions):
        model = models[train_region]
        for j, test_region in enumerate(regions):
            X_test = splits[test_region]["X_test"]
            y_test = splits[test_region]["y_test"]

            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            matrix[i][j] = rmse

    return matrix


def print_matrix(matrix):
    """Print the 4×4 RMSE matrix in a readable format."""
    regions = config.REGIONS

    # Header
    header = f"{'Train \\ Test':<15}"
    for r in regions:
        header += f"{r:>14}"
    print(header)
    print("-" * (15 + 14 * len(regions)))

    # Rows
    for i, train_region in enumerate(regions):
        row = f"{train_region:<15}"
        for j, test_region in enumerate(regions):
            val = matrix[i][j]
            if i == j:
                row += f"  {val:>10.4f} ✅"  # own-region
            else:
                row += f"  {val:>10.4f}   "   # cross-region
            
        print(row)


def analyze_results(matrix):
    """Analyze the RMSE matrix and draw conclusions."""
    regions = config.REGIONS

    print("\n--- Analysis ---")

    # Own-region RMSE (diagonal)
    own_rmses = [matrix[i][i] for i in range(len(regions))]
    avg_own = np.mean(own_rmses)

    # Cross-region RMSE (off-diagonal)
    cross_rmses = []
    for i in range(len(regions)):
        for j in range(len(regions)):
            if i != j:
                cross_rmses.append(matrix[i][j])
    avg_cross = np.mean(cross_rmses)

    print(f"\n  Average own-region RMSE (diagonal):    {avg_own:.4f}")
    print(f"  Average cross-region RMSE (off-diag):  {avg_cross:.4f}")
    print(f"  Degradation ratio (cross / own):       {avg_cross / avg_own:.2f}x")

    # Per-region analysis
    print(f"\n  Per-region breakdown:")
    for i, region in enumerate(regions):
        own = matrix[i][i]
        cross_for_this = [matrix[i][j] for j in range(len(regions)) if j != i]
        avg_cross_this = np.mean(cross_for_this)
        worst = max(cross_for_this)
        worst_region = regions[[j for j in range(len(regions)) if j != i][np.argmax(cross_for_this)]]
        print(f"    {region:<14} own={own:.4f}, avg cross={avg_cross_this:.4f} "
              f"({avg_cross_this/own:.2f}x), worst={worst:.4f} on {worst_region}")

    # Conclusion
    if avg_cross / avg_own > 1.2:
        print(f"\n  ✅ CONCLUSION: Cross-region RMSE is {avg_cross/avg_own:.2f}x higher than own-region.")
        print(f"     This proves regions have distinct signal behavior.")
        print(f"     A single model CANNOT generalize across all regions.")
    else:
        print(f"\n  ⚠️  Degradation ratio is only {avg_cross/avg_own:.2f}x — regions may not be as")
        print(f"     distinct as expected. Investigate further.")


def main():
    print("\n" + "=" * 60)
    print("PHASE 2: PROVE REGIONAL DISTINCTNESS")
    print("=" * 60)

    # Step 1: Load cleaned data
    print("\n[Step 1] Loading cleaned datasets...")
    datasets = load_cleaned_data()

    # Step 2: Train/test split
    print("\n[Step 2] Splitting data (80/20)...")
    splits = prepare_train_test_splits(datasets)

    # Step 3: Train models
    print("\n[Step 3] Training LightGBM models (default hyperparameters)...")
    models = train_models(splits)

    # Step 4: Cross-region RMSE matrix
    print("\n[Step 4] Building cross-region RMSE matrix...")
    matrix = build_cross_region_matrix(models, splits)

    # Step 5: Display results
    print("\n" + "=" * 60)
    print("CROSS-REGION RMSE MATRIX (log scale)")
    print("=" * 60)
    print()
    print_matrix(matrix)

    # Step 6: Analysis
    analyze_results(matrix)

    print("\n" + "=" * 60)
    print("✅ Phase 2 complete.")
    print("   Next: Phase 3 — PSO-Optimized Per-Region Models")
    print("=" * 60)


if __name__ == "__main__":
    main()
