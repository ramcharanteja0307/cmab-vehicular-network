"""
Phase 4: Neural Contextual Bandit (Neural-LinUCB)
=================================================
Implements the Neural-LinUCB algorithm to dynamically route traffic to the 
correct regional expert based ONLY on the 22 common features.

Architecture:
1. Deep Representation (PyTorch): Transforms the 22 common features into a 
   64-dimensional latent representation.
2. Shallow Exploration (LinUCB): Maintains uncertainty bounds and linear 
   weights for each of the 4 arms (Experts) on top of the latent representation.

Usage:
    python src/neural_bandit.py
"""

import os
# Prevent OpenMP segmentation fault on macOS (Apple Silicon) when mixing PyTorch & LightGBM
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import lightgbm as lgb
import pandas as pd
import numpy as np
import sys
import torch
# Disable PyTorch multithreading to completely avoid OpenMP deadlock on Mac
torch.set_num_threads(1)
import torch.nn as nn
import torch.optim as optim
import joblib
import warnings
import matplotlib.pyplot as plt

# Suppress LightGBM feature name warnings that freeze the terminal
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# =============================================================================
# 1. PyTorch Neural Network (Deep Representation)
# =============================================================================
class RepresentationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, x):
        return self.net(x)


# =============================================================================
# 2. Neural-LinUCB Agent
# =============================================================================
class NeuralLinUCB:
    def __init__(self, input_dim, latent_dim=32, num_arms=4, alpha=0.1, lr=1e-3):
        self.latent_dim = latent_dim
        self.num_arms = num_arms
        self.alpha = alpha  # Exploration parameter
        
        # Initialize Neural Network
        self.net = RepresentationNetwork(input_dim, 64, latent_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Initialize LinUCB parameters (A matrix and b vector for each arm)
        self.A_inv = [np.eye(latent_dim) for _ in range(num_arms)]
        self.b = [np.zeros(latent_dim) for _ in range(num_arms)]
        
        # Experience Replay Buffer for NN training
        self.buffer_x = []
        self.buffer_a = []
        self.buffer_r = []
        
    def select_arm(self, x_array, explore=True):
        """Select an arm based on UCB scores of the latent representation."""
        self.net.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_array).unsqueeze(0)
            # Get latent representation (z)
            z = self.net(x_tensor).numpy().flatten()
            z = z / np.sqrt(self.latent_dim)  # Normalize
            
        p = np.zeros(self.num_arms)
        for a in range(self.num_arms):
            # Calculate theta (linear weights)
            theta_a = self.A_inv[a].dot(self.b[a])
            
            if explore:
                exploration_bonus = self.alpha * np.sqrt(np.dot(z, self.A_inv[a].dot(z)))
            else:
                exploration_bonus = 0.0
                
            # Expected reward + exploration
            p[a] = np.dot(theta_a, z) + exploration_bonus
            
        chosen_arm = np.argmax(p)
        return chosen_arm, z
        
    def update(self, x_array, arm, reward, z):
        """Update LinUCB matrices and store experience."""
        # Sherman-Morrison formula for efficient A_inv update
        Az = self.A_inv[arm].dot(z)
        num = np.outer(Az, Az)
        den = 1.0 + np.dot(z, Az)
        self.A_inv[arm] -= num / den
        
        # Update target vector
        self.b[arm] += reward * z
        
        # Store experience
        self.buffer_x.append(x_array)
        self.buffer_a.append(arm)
        self.buffer_r.append(reward)
        
    def train_network(self, batch_size=64):
        """Train the Neural Network periodically."""
        if len(self.buffer_x) < batch_size:
            return
            
        self.net.train()
        indices = np.random.choice(len(self.buffer_x), batch_size, replace=False)
        batch_x = torch.FloatTensor(np.array([self.buffer_x[i] for i in indices]))
        batch_a = [self.buffer_a[i] for i in indices]
        batch_r = torch.FloatTensor([self.buffer_r[i] for i in indices])
        
        self.optimizer.zero_grad()
        z_batch = self.net(batch_x) / np.sqrt(self.latent_dim)
        
        # Predict reward using current linear parameters
        pred_r = torch.zeros(batch_size)
        for i in range(batch_size):
            a = batch_a[i]
            theta_a = torch.FloatTensor(self.A_inv[a].dot(self.b[a]))
            pred_r[i] = torch.dot(theta_a, z_batch[i])
            
        loss = self.criterion(pred_r, batch_r)
        loss.backward()
        self.optimizer.step()


# =============================================================================
# 3. Data Stream Simulation
# =============================================================================
def load_and_prepare_stream():
    """Combines all datasets into a single shuffled data stream."""
    print("  Loading cleaned datasets and expert models...")
    dataframes = []
    experts = {}
    expert_features = {}
    
    for idx, region in enumerate(config.REGIONS):
        # Load Data
        df = pd.read_csv(config.CLEANED_FILES[region])
        
        # Precompute the local features this expert expects
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in [config.TARGET_RAW, config.TARGET_LOG, config.TARGET_NORM, 'true_arm']]
        expert_features[idx] = feature_cols
        
        df['true_arm'] = idx
        df['region_name'] = region
        dataframes.append(df)
        
        # Load Expert Model
        model_path = os.path.join(config.MODELS_DIR, f"lgbm_expert_{region}.joblib")
        experts[idx] = joblib.load(model_path)
        
    # Combine and shuffle
    stream = pd.concat(dataframes, ignore_index=True)
    stream = stream.sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)
    
    print(f"  Created mixed data stream with {len(stream):,} rows.")
    return stream, experts, expert_features


def run_simulation():
    print("\n" + "=" * 60)
    print("PHASE 4: NEURAL CONTEXTUAL BANDIT SIMULATION")
    print("=" * 60)
    
    stream, experts, expert_features = load_and_prepare_stream()
    
    # Standardize the 22 common features for the Neural Network
    scaler = StandardScaler()
    X_common = scaler.fit_transform(stream[config.COMMON_FEATURES])
    
    # Initialize Agent
    agent = NeuralLinUCB(input_dim=len(config.COMMON_FEATURES), num_arms=4)
    
    total_steps = len(stream)
    train_size = int(total_steps * 0.8)
    
    # Tracking metrics for learning curve
    history_t = []
    history_acc = []
    rolling_correct = 0
    
    print(f"\n  [Phase A] Training Bandit on 80% of data ({train_size:,} rows)...")
    for t in range(train_size):
        x_t = X_common[t]
        true_arm = stream.loc[t, 'true_arm']
        true_y = stream.loc[t, config.TARGET_NORM]
        
        # explore=True allows the bandit to try new arms and learn
        selected_arm, z_t = agent.select_arm(x_t, explore=True)
        
        if selected_arm == true_arm:
            rolling_correct += 1
            expert = experts[selected_arm]
            feature_cols = expert_features[selected_arm]
            row = stream.iloc[[t]]
            y_pred = expert.predict(row[feature_cols])[0]
            reward = 1.0 - abs(true_y - y_pred)
        else:
            reward = 0.0
            
        # 4. Agent learns from the reward
        agent.update(x_t, selected_arm, reward, z_t)
        
        # Train Neural Network periodically
        if t % 50 == 0:
            agent.train_network()
            
        # Log accuracy every 5000 steps to show learning progress
        if (t + 1) % 5000 == 0:
            acc = (rolling_correct / 5000) * 100
            history_t.append(t + 1)
            history_acc.append(acc)
            print(f"    Training timestep {t+1:,} / {train_size:,} | Recent Accuracy: {acc:.1f}%")
            rolling_correct = 0

    # Save Learning Curve Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history_t, history_acc, marker='o', color='b', linewidth=2)
    plt.title('Neural Bandit Learning Curve (Phase A Training)')
    plt.xlabel('Training Steps')
    plt.ylabel('Routing Accuracy (%)')
    plt.grid(True)
    plt.savefig('learning_curve.png')
    print("  ✅ Saved learning curve plot to 'learning_curve.png'")
    
    # Save the trained Bandit Brain
    print("\n  Saving trained Bandit brain to 'models/'...")
    torch.save(agent.net.state_dict(), os.path.join(config.MODELS_DIR, 'bandit_net.pth'))
    np.save(os.path.join(config.MODELS_DIR, 'bandit_A_inv.npy'), agent.A_inv)
    np.save(os.path.join(config.MODELS_DIR, 'bandit_b.npy'), agent.b)
    joblib.dump(scaler, os.path.join(config.MODELS_DIR, 'bandit_scaler.joblib'))

    print(f"\n  [Phase B] Evaluating Bandit on remaining 20% ({total_steps - train_size:,} rows)...")
    correct_routing = 0
    cumulative_rmse = 0.0
    
    for t in range(train_size, total_steps):
        if t % 10000 == 0:
            print(f"    Evaluating timestep {t:,} / {total_steps:,}...")
            
        x_t = X_common[t]
        true_arm = stream.loc[t, 'true_arm']
        true_y = stream.loc[t, config.TARGET_NORM]
        
        # explore=False turns off learning and just uses pure exploitation/prediction
        selected_arm, z_t = agent.select_arm(x_t, explore=False)
        
        if selected_arm == true_arm:
            correct_routing += 1
            expert = experts[selected_arm]
            feature_cols = expert_features[selected_arm]
            row = stream.iloc[[t]]
            y_pred = expert.predict(row[feature_cols])[0]
            error = abs(true_y - y_pred)
            cumulative_rmse += error**2
        else:
            cumulative_rmse += 1.0 # Max error penalty
            
    test_steps = total_steps - train_size
    final_rmse = np.sqrt(cumulative_rmse / test_steps)
    accuracy = (correct_routing / test_steps) * 100
    
    print("\n" + "=" * 60)
    print("FINAL SIMULATION RESULTS (EVALUATED ON 20% UNSEEN DATA)")
    print("=" * 60)
    print(f"  Training Steps:      {train_size:,}")
    print(f"  Test Steps:          {test_steps:,}")
    print(f"  Test Accuracy:       {accuracy:.2f}%")
    print(f"  Test System RMSE:    {final_rmse:.4f}")
    
    if accuracy > 80:
        print("\n  ✅ SUCCESS! The Neural Bandit generalizes perfectly to")
        print("     unseen environments without memorizing.")
    else:
        print("\n  ⚠️ Accuracy is below 80%. More feature engineering or")
        print("     network tuning may be required.")
        
    print("=" * 60)


if __name__ == "__main__":
    run_simulation()
