# Contextual Multi-Armed Bandit for Vehicular Networks

This repository contains a publication-ready AI pipeline that implements a **Neural-LinUCB** architecture (Deep Representation + Shallow Exploration). It dynamically routes vehicular network signals to specialized regional expert models (Highway, Avenue, Park, Residential) based on real-time context.

The architecture ensures extremely high prediction accuracy by allowing highly tuned local experts to make the final predictions, while a centralized Neural Bandit mathematically handles the exploration and routing.

##  Key Results
- **Routing Accuracy:** > 92.00% on strictly unseen data.
- **System RMSE:** Approaches the accuracy of dedicated regional experts without the risk of out-of-distribution performance degradation.
- *Check the `plots/` directory for the 4x4 Generalization Heatmap and the Bandit Learning Curve.*

---

## 🛠️ Setup & Installation

**1. Clone the repository:**
```bash
git clone <your-repository-url>
cd cmab-vehicular-network
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```
*(Note for macOS Apple Silicon users: The pipeline automatically handles OpenMP threading locks between PyTorch and LightGBM).*

---

## 📖 Step-by-Step Execution Guide

This pipeline is designed to be run sequentially from Phase 1 to Phase 5.

### Phase 1: Data Preparation
Cleans the raw datasets, removes strictly defined data-leakage features (to prevent cheating), log-transforms the data rate, and applies global MinMax normalization.
```bash
python3 src/data_preparation.py
```

### Phase 2: Prove Regional Distinctness
Scientifically justifies the need for a routing bandit. It trains standard models on each region and tests them across all other regions to generate a 4x4 RMSE matrix, proving that cross-region prediction causes massive performance degradation.
```bash
python3 src/prove_regional_distinctness.py
```

### Phase 3: Train PSO-Optimized Experts
Uses Particle Swarm Optimization (PSO) to tune the hyperparameters of 4 distinct LightGBM models. These act as the final "Arms" of the bandit.
```bash
python3 src/pso_regional_models.py
```

### Phase 4: Train the Neural Contextual Bandit
Simulates real-world online learning. A PyTorch network compresses the 22 common context features into a latent representation. The LinUCB algorithm explores/exploits this representation to route traffic to the best expert.
```bash
python3 src/neural_bandit.py
```
*(This script will perform an 80/20 train/test split, calculate the final routing accuracy, and save the Bandit's "Brain" into the `models/` directory).*

### Phase 5: Test on Completely Unseen Data
Test the fully trained pipeline on your own custom dataset. The script will load the frozen Bandit brain and the 4 regional experts.
```bash
python3 src/predict_unseen.py /path/to/your/custom_dataset.csv
```
**Notes for custom data:**
- The CSV will automatically be preprocessed using the saved Scaler.
- Missing local features will be gracefully handled.
- If your CSV includes a `region` column, the script will automatically calculate the Routing Accuracy %.
- The script generates a `_results.csv` file with the Bandit's decisions and final data rate predictions appended.

---

## Generating Publication Plots
If you want to regenerate the beautiful Seaborn heatmap showing the 4x4 Cross-Region RMSE Matrix using the final PSO-optimized models, run:
```bash
python3 src/generate_plots.py
```

##  Architecture Details
- **Context Vector ($x_t$):** 22 guaranteed common features (e.g., speed, cell ID, basic signal strength).
- **Deep Representation ($z_t$):** 3-layer PyTorch Neural Network compressing the 22 features into a 32-dimensional latent representation.
- **Shallow Exploration:** LinUCB algorithm mathematically deciding which expert (Arm) has the highest probability of minimizing the prediction error based on $z_t$.
- **Experts:** 4 LightGBM models trained via PSO on region-specific features.
