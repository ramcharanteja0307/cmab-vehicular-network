import os
# Prevent OpenMP segmentation fault on macOS (Apple Silicon) when mixing PyTorch & LightGBM
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import lightgbm as lgb
import torch
torch.set_num_threads(1)

import pandas as pd
import numpy as np
import sys
import torch.nn as nn
import joblib
import warnings
import matplotlib.pyplot as plt

# Suppress LightGBM feature name warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from neural_bandit import RepresentationNetwork

def load_brain_and_experts():
    """Loads the completely frozen brain and all experts."""
    print("  Loading Frozen Bandit Brain...")
    
    # Load Scaler
    scaler = joblib.load(os.path.join(config.MODELS_DIR, 'bandit_scaler.joblib'))
    
    # Load Neural Network
    net = RepresentationNetwork(len(config.COMMON_FEATURES), 64, 32)
    net.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, 'bandit_net.pth')))
    net.eval()
    
    # Load LinUCB Matrices
    A_inv = np.load(os.path.join(config.MODELS_DIR, 'bandit_A_inv.npy'))
    b = np.load(os.path.join(config.MODELS_DIR, 'bandit_b.npy'))
    
    print("  Loading 4 Regional Experts...")
    experts = {}
    expert_features = {}
    for idx, region in enumerate(config.REGIONS):
        model_path = os.path.join(config.MODELS_DIR, f"lgbm_expert_{region}.joblib")
        experts[idx] = joblib.load(model_path)
        
        # We also need to know what features this expert expects.
        # We can look at the columns of the cleaned files.
        df_dummy = pd.read_csv(config.CLEANED_FILES[region], nrows=1)
        numeric_cols = df_dummy.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in [config.TARGET_RAW, config.TARGET_LOG, config.TARGET_NORM, 'true_arm', 'region_name']]
        expert_features[idx] = feature_cols
        
    return scaler, net, A_inv, b, experts, expert_features

def test_unseen_data(csv_path):
    print("\n" + "=" * 60)
    print("TESTING COMPLETELY UNSEEN DATA")
    print("=" * 60)
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: File {csv_path} does not exist.")
        return
        
    print(f"  Loading unseen dataset: {csv_path}")
    stream = pd.read_csv(csv_path)
    
    # Check for labels and region column
    has_labels = config.TARGET_NORM in stream.columns
    has_region = 'region' in stream.columns or 'true_arm' in stream.columns
    
    if 'region' in stream.columns and 'true_arm' not in stream.columns:
        region_map = {r: i for i, r in enumerate(config.REGIONS)}
        stream['true_arm'] = stream['region'].str.lower().map(region_map)
        
    scaler, net, A_inv, b, experts, expert_features = load_brain_and_experts()
    
    print("\n  Extracting Context Features...")
    # Fill any missing common features with 0 just in case
    for col in config.COMMON_FEATURES:
        if col not in stream.columns:
            stream[col] = 0
            
    X_common = scaler.transform(stream[config.COMMON_FEATURES])
    
    predictions = []
    routed_regions = []
    
    correct_routing = 0
    cumulative_rmse = 0.0
    
    print(f"  Running {len(stream):,} rows through the frozen Bandit...")
    for t in range(len(stream)):
        if t % 5000 == 0 and t > 0:
            print(f"    Processed {t:,} rows...")
            
        x_t = X_common[t]
        
        # Pass through Neural Network
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_t).unsqueeze(0)
            z = net(x_tensor).numpy().flatten()
            z = z / np.sqrt(32)  # normalize
            
        # Select Arm (Strictly No Exploration - pure exploitation)
        p = np.zeros(4)
        for a in range(4):
            theta_a = A_inv[a].dot(b[a])
            p[a] = np.dot(theta_a, z)
            
        selected_arm = np.argmax(p)
        routed_regions.append(config.REGIONS[selected_arm])
        
        # Let the expert predict
        expert = experts[selected_arm]
        feature_cols = expert_features[selected_arm]
        
        # If the unseen dataset is missing some local features, fill them with NaNs 
        row_data = stream.iloc[[t]].copy()
        for col in feature_cols:
            if col not in row_data.columns:
                row_data[col] = np.nan
                
        y_pred = expert.predict(row_data[feature_cols])[0]
        predictions.append(y_pred)
        
        # Calculate Routing Accuracy if we have the region
        if has_region:
            if selected_arm == stream.loc[t, 'true_arm']:
                correct_routing += 1
                
        # Calculate RMSE if we have the normalized data rate labels
        if has_labels:
            true_y = stream.loc[t, config.TARGET_NORM]
            error = abs(true_y - y_pred)
            cumulative_rmse += error**2
            
    stream['Bandit_Predicted_Region'] = routed_regions
    stream['Bandit_Predicted_Datarate'] = predictions
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    
    # Print the distribution of where the Bandit sent the traffic
    print("\n  Traffic Routing Distribution:")
    routing_counts = stream['Bandit_Predicted_Region'].value_counts()
    for region, count in routing_counts.items():
        print(f"    {region.capitalize()}: {count:,} rows ({(count/len(stream))*100:.1f}%)")
        
    if has_region:
        accuracy = (correct_routing / len(stream)) * 100
        print(f"\n  🎯 Routing Accuracy: {accuracy:.2f}%")
        
    if has_labels:
        final_rmse = np.sqrt(cumulative_rmse / len(stream))
        print(f"  📉 Unseen Data System RMSE: {final_rmse:.4f}")
    
    # Save the results to a new CSV
    output_path = csv_path.replace('.csv', '_results.csv')
    stream.to_csv(output_path, index=False)
    print(f"\n  ✅ Saved complete predictions to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 src/predict_unseen.py <path_to_unseen_csv>")
        print("Example: python3 src/predict_unseen.py /Users/apple/Downloads/1779200653147_residential_cleaned.csv")
        sys.exit(1)
        
    test_unseen_data(sys.argv[1])
