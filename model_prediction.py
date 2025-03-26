import os
import numpy as np
import librosa
import joblib
from audio_utils import load_audio
from feature_extraction import extract_features

# Load Model, Scaler, and Feature Selector
model = joblib.load("models/emotion_classifier.pkl")
scaler = joblib.load("models/scaler.pkl")
selected_indices = joblib.load("models/selected_feature_indices.pkl")

# Emotion Labels
EMOTIONS = {
    0: "Neutral", 1: "Calm", 2: "Happy", 3: "Sad",
    4: "Angry", 5: "Fearful", 6: "Disgust", 7: "Surprised"
}

def predict_emotion(audio_path):
    """Predicts emotion for a given audio file."""
    y, sr = load_audio(audio_path)
    features = extract_features(y, sr)

    if features is None or len(features) == 0:
        return {"file": os.path.basename(audio_path), "error": "Feature extraction failed."}

    features = scaler.transform([features])  
    features_selected = features[:, selected_indices]

    prediction = model.predict(features_selected)[0]
    return {"file": os.path.basename(audio_path), "emotion": EMOTIONS.get(prediction, "Unknown")}

def predict_batch_emotions(directory_path):
    """Predict emotions for all audio files in a given directory and its subdirectories."""
    results = []
    
    # Loop through all the actor directories and their audio files
    for actor_folder in os.listdir(directory_path):
        actor_folder_path = os.path.join(directory_path, actor_folder)
        if os.path.isdir(actor_folder_path):  # Check if it's a directory
            for audio_file in os.listdir(actor_folder_path):
                if audio_file.endswith(".wav"):  # Ensure the file is a .wav file
                    audio_path = os.path.join(actor_folder_path, audio_file)
                    result = predict_emotion(audio_path)
                    results.append(result)
    
    return results

if __name__ == "__main__":
    # Provide the path to the root directory where actor folders are stored
    root_directory = "data/raw"  # Path to your 'data/raw' folder
    
    # Get predictions for all audio files inside the 'data/raw' folder
    results = predict_batch_emotions(root_directory)
    
    # Print the results for each file
    for result in results:
        print(result)
