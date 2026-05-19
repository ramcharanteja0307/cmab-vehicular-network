"""
Configuration file for CMAB Vehicular Network Prediction Project.
Single source of truth for all paths, feature sets, and hyperparameters.
"""

import os

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
CLEANED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

# =============================================================================
# REGIONS (each region = one bandit arm)
# =============================================================================
REGIONS = ["highway", "avenue", "park", "residential"]

RAW_FILES = {
    "highway": os.path.join(RAW_DATA_DIR, "highway_data.csv"),
    "avenue": os.path.join(RAW_DATA_DIR, "avenue_data.csv"),
    "park": os.path.join(RAW_DATA_DIR, "park_data.csv"),
    "residential": os.path.join(RAW_DATA_DIR, "residential_data.csv"),
}

CLEANED_FILES = {
    region: os.path.join(CLEANED_DATA_DIR, f"{region}_cleaned.csv")
    for region in REGIONS
}

# =============================================================================
# TARGET VARIABLE
# =============================================================================
TARGET_RAW = "datarate"
TARGET_LOG = "log_datarate"  # log(datarate)
TARGET_NORM = "log_datarate_norm"  # MinMax normalized log(datarate) — used for all modeling

# =============================================================================
# COLUMNS TO DROP (with reasons)
# =============================================================================
DROP_COLUMNS = {
    # --- Data Leakage ---
    "measured_qos":     "Binary flag derived from datarate — circular dependency",
    "target_datarate":  "Test configuration target — not available in real-time (Highway only)",
    
    # --- Experimental Metadata ---
    "scenario":         "Experiment scenario ID (Highway only)",
    "direction":        "Driving direction label (Highway only)",
    "operator":         "Network operator label (Highway only)",
    
    # --- No Information ---
    "area":             "Constant value 0 across all datasets — zero information",
    
    # --- Non-numeric / Identifiers ---
    "timestamp":        "Already decomposed into hour and minute columns",
    "device":           "Measurement device ID — not a predictive feature",
    "Traffic Street Name": "Categorical street identifier — not useful for models",
    
    # --- Leaky Network Metrics (only in Highway, directly proportional to datarate) ---
    "PCell_Downlink_TB_Size":  "Transport block size — directly proportional to datarate",
    "PCell_Uplink_TB_Size":    "Uplink transport block size — same leakage concern",
    "measurement":             "Measurement counter (Highway only)",
}

# NOTE on features we KEEP:
# - PCell_Downlink_Num_RBs: Network scheduler allocates resource blocks based on
#   channel quality (RSRP/RSRQ/CQI), NOT based on achieved datarate. It's a CAUSE.
# - drive_mode: Represents driving behavior mode, used in reference paper.

# =============================================================================
# COMMON FEATURES (present in all 4 datasets after dropping)
# Used as input to the Neural Bandit (representation learning)
# =============================================================================
COMMON_FEATURES = [
    # --- Experimental Metadata ---
    "drive_mode",                  # Experimental setup label (0/1/2)
    # --- Network Signal ---
    "PCell_RSRP_max",              # Signal strength (dBm)
    "PCell_RSRQ_max",              # Signal quality (dB)
    "PCell_E-ARFCN",               # Frequency channel number
    "PCell_Downlink_Average_MCS",   # Modulation and coding scheme
    "PCell_Uplink_Num_RBs",         # Uplink resource blocks allocated
    "PCell_Uplink_Tx_Power_(dBm)",  # Uplink transmit power
    "PCell_Cell_ID",               # Serving cell ID
    "PCell_TAC",                   # Tracking area code
    
    # --- Network Performance ---
    "ping_ms",                     # Latency
    "jitter",                      # Signal instability
    
    # --- Mobility ---
    "speed_kmh",                   # Vehicle speed
    "Latitude",                    # GPS latitude
    "Altitude",                    # GPS altitude
    "COG",                         # Course over ground (heading)
    
    # --- Environment ---
    "temperature",                 # Weather
    "windSpeed",                   # Weather
    "uvIndex",                     # Weather / time-of-day proxy
    "Traffic Jam Factor",          # Traffic density
    "Traffic Distance",            # Distance in traffic
    
    # --- Time ---
    "hour",                        # Hour of day
    "minute",                      # Minute of hour
]
# Total: 21 features

# =============================================================================
# RANDOM SEED (reproducibility)
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# TRAIN/TEST SPLIT
# =============================================================================
TEST_SIZE = 0.2
