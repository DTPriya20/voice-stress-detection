import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# 🔹 Load extracted features
FEATURES_FILE = "data/extracted_features.npy"
SELECTED_INDICES_FILE = "models/selected_feature_indices.pkl"
SCALER_FILE = "models/scaler.pkl"

if not os.path.exists(FEATURES_FILE):
    raise FileNotFoundError(f"❌ Extracted features file not found: {FEATURES_FILE}")

if not os.path.exists(SELECTED_INDICES_FILE):
    raise FileNotFoundError(f"❌ Selected indices file not found: {SELECTED_INDICES_FILE}")

# ✅ Load data
X_all = np.load(FEATURES_FILE)
selected_indices = joblib.load(SELECTED_INDICES_FILE)

# ✅ Ensure correct feature selection
if X_all.shape[1] != len(selected_indices):
    raise ValueError(f"❌ Feature count mismatch! Expected {len(selected_indices)}, but got {X_all.shape[1]}.")

# ✅ Standardize selected features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# ✅ Save the trained scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, SCALER_FILE)

print(f"✅ Scaler retrained and saved as {SCALER_FILE}")
