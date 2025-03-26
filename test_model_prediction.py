import os
import numpy as np
import joblib

# 🔹 File paths
FEATURES_FILE = "data/extracted_features.npy"
SCALER_FILE = "models/scaler.pkl"
MODEL_FILE = "models/trained_model.pkl"

# ✅ Check if files exist
if not os.path.exists(FEATURES_FILE):
    raise FileNotFoundError(f"❌ Extracted features file not found: {FEATURES_FILE}")

if not os.path.exists(SCALER_FILE):
    raise FileNotFoundError(f"❌ Scaler file not found: {SCALER_FILE}")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_FILE}")

# ✅ Load extracted features
X_all = np.load(FEATURES_FILE)

# ✅ Load scaler and scale features
scaler = joblib.load(SCALER_FILE)
X_scaled = scaler.transform(X_all)

# ✅ Load trained model
model = joblib.load(MODEL_FILE)

# ✅ Make predictions
predictions = model.predict(X_scaled)

# ✅ Display results
print(f"✅ Model successfully made predictions on {X_scaled.shape[0]} samples.")
print(f"📊 First 5 Predictions: {predictions[:5]}")
