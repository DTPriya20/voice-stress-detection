import os
import numpy as np
import librosa
import joblib

# 🔹 Path to the dataset
DATASET_PATH = "data/raw"

# 🔹 Load corrected feature indices
SELECTED_INDICES_FILE = "models/selected_feature_indices.pkl"

if not os.path.exists(SELECTED_INDICES_FILE):
    raise FileNotFoundError(f"❌ {SELECTED_INDICES_FILE} not found!")

selected_indices = joblib.load(SELECTED_INDICES_FILE)

# 🔹 Function to extract features from an audio file
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Extract 40 MFCC features
        mfccs_mean = np.mean(mfccs, axis=1)  # Compute mean of each feature
        
        # ✅ Select only valid feature indices
        selected_features = mfccs_mean[selected_indices]
        
        return selected_features
    except Exception as e:
        print(f"❌ Feature extraction failed for {file_path}: {e}")
        return None

# 🔹 Loop through all audio files
all_features = []
for actor_folder in sorted(os.listdir(DATASET_PATH)):
    actor_path = os.path.join(DATASET_PATH, actor_folder)
    if os.path.isdir(actor_path):  # Ensure it's a folder
        for file in sorted(os.listdir(actor_path)):
            if file.endswith(".wav"):
                file_path = os.path.join(actor_path, file)
                features = extract_features(file_path)

                if features is not None:
                    all_features.append(features)

# 🔹 Convert extracted features to NumPy array
X_all = np.array(all_features)

# 🔹 Save the extracted features
os.makedirs("data", exist_ok=True)
np.save("data/extracted_features.npy", X_all)

print(f"✅ Feature extraction complete! Saved {X_all.shape[0]} samples with {X_all.shape[1]} features.")
