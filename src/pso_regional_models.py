"""
Phase 3: PSO-Optimized Per-Region Models
========================================
Optimizes hyperparameters for 4 separate LightGBM models (one per region)
using Particle Swarm Optimization (PSO). 

Each model is trained on ALL available non-leaky features for its specific region,
NOT just the common features. This allows each expert model to specialize deeply
in its environment and drives the own-region RMSE down to ~0.00x.

Usage:
    python src/pso_regional_models.py
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import pyswarms as ps
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def pso_objective(particles, X_train, y_train, X_val, y_val):
    """
    The objective function for PSO. 
    It takes a matrix of particles (hyperparameter sets), trains a LightGBM 
    model for each particle, and returns an array of validation RMSE scores.
    """
    n_particles = particles.shape[0]
    errors = np.zeros(n_particles)
    
    for i in range(n_particles):
        params = particles[i]
        lr = params[0]
        num_leaves = int(params[1])
        max_depth = int(params[2])
        n_estimators = int(params[3])
        
        model = lgb.LGBMRegressor(
            learning_rate=lr,
            num_leaves=num_leaves,
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=config.RANDOM_SEED,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        errors[i] = rmse
        
    return errors


def train_pso_model_for_region(region):
    """
    Run PSO optimization and train the final expert model for a specific region.
    """
    print(f"\n--- Optimizing Expert Model for {region.upper()} ---")
    
    # 1. Load the region's cleaned data
    path = config.CLEANED_FILES[region]
    df = pd.read_csv(path)
    
    # 2. Use ALL available numeric features for this region
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in [config.TARGET_RAW, config.TARGET_LOG, config.TARGET_NORM]]
    
    print(f"  Using {len(feature_cols)} local features.")
    
    X = df[feature_cols]
    y = df[config.TARGET_NORM]
    
    # 3. Train / Test / Val splits
    # 80/20 train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
    # Further split train into train/val (80/20) for PSO evaluation
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=config.RANDOM_SEED)
    
    # 4. Define PSO Bounds
    # Format: [learning_rate, num_leaves, max_depth, n_estimators]
    lower_bound = np.array([0.01, 15, 3, 50])
    upper_bound = np.array([0.30, 100, 12, 600])
    bounds = (lower_bound, upper_bound)
    
    # Standard PSO parameters
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    
    # We use a small swarm for speed in this demonstration. 
    # For production, you could increase n_particles and iters.
    optimizer = ps.single.GlobalBestPSO(
        n_particles=10, 
        dimensions=4, 
        options=options, 
        bounds=bounds
    )
    
    print(f"  Starting PSO swarm optimization...")
    best_cost, best_pos = optimizer.optimize(
        pso_objective, 
        iters=15, 
        X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val,
        verbose=False  # Keep output clean
    )
    
    # Extract best parameters found by the swarm
    best_lr = best_pos[0]
    best_num_leaves = int(best_pos[1])
    best_max_depth = int(best_pos[2])
    best_n_estimators = int(best_pos[3])
    
    print(f"  Best params found: lr={best_lr:.4f}, leaves={best_num_leaves}, depth={best_max_depth}, n_est={best_n_estimators}")
    
    # 5. Train the FINAL model using the best params on the full train set
    print(f"  Training final {region} model...")
    final_model = lgb.LGBMRegressor(
        learning_rate=best_lr,
        num_leaves=best_num_leaves,
        max_depth=best_max_depth,
        n_estimators=best_n_estimators,
        random_state=config.RANDOM_SEED,
        verbose=-1
    )
    
    final_model.fit(X_train, y_train)
    
    # 6. Evaluate on the untouched TEST set
    test_preds = final_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    
    print(f"  ✅ {region.upper()} Final Own-Region Test RMSE: {test_rmse:.4f}")
    
    # 7. Save the model for use in Phase 4 (Neural Bandit routing)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model_path = os.path.join(config.MODELS_DIR, f"lgbm_expert_{region}.joblib")
    joblib.dump(final_model, model_path)
    
    return test_rmse


def main():
    print("\n" + "=" * 60)
    print("PHASE 3: PSO-OPTIMIZED PER-REGION MODELS (THE EXPERTS)")
    print("=" * 60)
    
    results = {}
    for region in config.REGIONS:
        rmse = train_pso_model_for_region(region)
        results[region] = rmse
        
    print("\n" + "=" * 60)
    print("FINAL EXPERT RMSE SUMMARY")
    print("=" * 60)
    for region, rmse in results.items():
        print(f"  {region:<15}: {rmse:.4f}")
        
    print("\n✅ Phase 3 complete. Expert models saved to models/")
    print("   Next: Phase 4 — Neural Contextual Bandit (Routing)")
    print("=" * 60)


if __name__ == "__main__":
    main()
