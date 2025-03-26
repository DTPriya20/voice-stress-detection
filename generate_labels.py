import os
import numpy as np

DATASET_PATH = "data/raw"
LABELS_FILE = "data/labels.npy"

# 🎭 Mapping emotion codes to labels
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

labels = []

for actor_folder in sorted(os.listdir(DATASET_PATH)):
    actor_path = os.path.join(DATASET_PATH, actor_folder)
    if os.path.isdir(actor_path):  
        for file in sorted(os.listdir(actor_path)):
            if file.endswith(".wav"):
                emotion_code = file.split("-")[2]  # Extract emotion from filename
                emotion_label = EMOTION_MAP.get(emotion_code, "unknown")
                labels.append(emotion_label)

# Convert labels to NumPy array and save
np.save(LABELS_FILE, np.array(labels))
print(f"✅ Labels generated and saved as {LABELS_FILE}")
