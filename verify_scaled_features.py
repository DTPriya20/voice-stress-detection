import os
import numpy as np
import joblib

# 🔹 File paths
FEATURES_FILE = "data/extracted_features.npy"
SCALER_FILE = "models/scaler.pkl"

# ✅ Check if files exist
if not os.path.exists(FEATURES_FILE):
    raise FileNotFoundError(f"❌ Extracted features file not found: {FEATURES_FILE}")

if not os.path.exists(SCALER_FILE):
    raise FileNotFoundError(f"❌ Scaler file not found: {SCALER_FILE}")

# ✅ Load extracted features
X_all = np.load(FEATURES_FILE)

# ✅ Load scaler
scaler = joblib.load(SCALER_FILE)

# ✅ Transform features using the trained scaler
X_scaled = scaler.transform(X_all)

# ✅ Print verification details
print(f"✅ Loaded {X_scaled.shape[0]} samples with {X_scaled.shape[1]} scaled features.")
print(f"📊 Sample of first 5 scaled feature vectors:\n{X_scaled[:5]}")
