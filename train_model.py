import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 🔹 File paths
FEATURES_FILE = "data/extracted_features.npy"
SCALER_FILE = "models/scaler.pkl"
MODEL_FILE = "models/trained_model.pkl"

# ✅ Check if necessary files exist
if not os.path.exists(FEATURES_FILE):
    raise FileNotFoundError(f"❌ Extracted features file not found: {FEATURES_FILE}")

if not os.path.exists(SCALER_FILE):
    raise FileNotFoundError(f"❌ Scaler file not found: {SCALER_FILE}")

# ✅ Load extracted features
X_all = np.load(FEATURES_FILE)

# ✅ Load labels (For now, assuming binary labels 0 & 1)
# Replace this with your actual labels file
LABELS_FILE = "data/labels.npy"
if not os.path.exists(LABELS_FILE):
    raise FileNotFoundError(f"❌ Labels file not found: {LABELS_FILE}")

y_all = np.load(LABELS_FILE)

# ✅ Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# ✅ Load and apply scaler
scaler = joblib.load(SCALER_FILE)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ✅ Save trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_FILE)

# ✅ Evaluate model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model training complete!")
print(f"📊 Accuracy on test set: {accuracy:.4f}")
print(f"📂 Model saved as: {MODEL_FILE}")
