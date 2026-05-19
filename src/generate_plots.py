import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to the path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def generate_heatmap():
    print("Generating 4x4 Cross-Region RMSE Heatmap...")
    
    # 1. Load the 4 PSO-Optimized Experts
    experts = {}
    for region in config.REGIONS:
        model_path = os.path.join(config.MODELS_DIR, f"lgbm_expert_{region}.joblib")
        experts[region] = joblib.load(model_path)
        
    # 2. Load the cleaned datasets
    datasets = {}
    for region in config.REGIONS:
        datasets[region] = pd.read_csv(config.CLEANED_FILES[region])
        
    # 3. Calculate the 4x4 RMSE matrix
    matrix = np.zeros((4, 4))
    
    for i, model_region in enumerate(config.REGIONS):
        model = experts[model_region]
        
        for j, data_region in enumerate(config.REGIONS):
            df = datasets[data_region]
            y_true = df[config.TARGET_NORM]
            
            # The expert expects specific features. If the data is from another region,
            # it might lack some features. We pad missing with NaN.
            df_dummy = pd.read_csv(config.CLEANED_FILES[model_region], nrows=1)
            numeric_cols = df_dummy.select_dtypes(include=[np.number]).columns
            feature_cols = [c for c in numeric_cols if c not in [config.TARGET_RAW, config.TARGET_LOG, config.TARGET_NORM, 'true_arm', 'region_name']]
            
            # Prepare test data with exact columns the model expects
            X_test = pd.DataFrame(index=df.index, columns=feature_cols)
            for col in feature_cols:
                if col in df.columns:
                    X_test[col] = df[col]
                else:
                    X_test[col] = np.nan
                    
            y_pred = model.predict(X_test)
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            matrix[i, j] = rmse
            
    # 4. Plot the Heatmap
    plt.figure(figsize=(8, 6))
    
    # Format region names for the axis
    labels = [r.capitalize() for r in config.REGIONS]
    
    ax = sns.heatmap(matrix, annot=True, fmt=".4f", cmap="YlOrRd", 
                     xticklabels=labels, yticklabels=labels,
                     cbar_kws={'label': 'RMSE Error'})
    
    plt.title("Cross-Region Generalization Error (RMSE)", pad=20, fontsize=14, fontweight='bold')
    plt.xlabel("Testing Environment (Data)", labelpad=15, fontsize=12)
    plt.ylabel("Trained Expert Model", labelpad=15, fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('plots', exist_ok=True)
    plot_path = os.path.join('plots', 'cross_region_rmse_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved heatmap to {plot_path}")


if __name__ == "__main__":
    generate_heatmap()
