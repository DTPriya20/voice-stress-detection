import librosa
import numpy as np
import os
from audio_utils import load_audio

def extract_features(y, sr):
    """
    Extract audio features from an audio signal.
    
    Parameters:
    y (np.array): Audio time series
    sr (int): Sampling rate
    
    Returns:
    np.array: Feature vector
    """
    if y is None or sr is None:
        print("[ERROR] Invalid audio input.")
        return None

    features = []

    try:
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        if mfccs.shape[1] > 0:
            features.extend(np.mean(mfccs, axis=1))

        # MFCC first and second derivatives
        mfccs_delta = librosa.feature.delta(mfccs)
        if mfccs_delta.shape[1] > 0:
            features.extend(np.mean(mfccs_delta, axis=1))

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        if spectral_centroid.shape[1] > 0:
            features.extend(np.mean(spectral_centroid, axis=1))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        if spectral_bandwidth.shape[1] > 0:
            features.extend(np.mean(spectral_bandwidth, axis=1))

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        if spectral_contrast.shape[1] > 0:
            features.extend(np.mean(spectral_contrast, axis=1))

        # Tonal features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        if chroma.shape[1] > 0:
            features.extend(np.mean(chroma, axis=1))

        # Rhythm features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(float(tempo))

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        if zcr.shape[1] > 0:
            features.extend(np.mean(zcr, axis=1))

        # Root Mean Square Energy
        rms = librosa.feature.rms(y=y)
        if rms.shape[1] > 0:
            features.extend(np.mean(rms, axis=1))

    except Exception as e:
        print(f"[ERROR] Feature extraction error: {e}")
        return None

    return np.array(features, dtype=np.float32)

def find_audio_files(directory):
    """
    Recursively find all audio files inside a directory.
    
    Parameters:
    directory (str): Root directory containing audio files
    
    Returns:
    list: List of file paths
    """
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):  # Add other formats if needed
                audio_files.append(os.path.join(root, file))
    return audio_files

def extract_features_batch(directory):
    """
    Extract features from all audio files in a directory.
    
    Parameters:
    directory (str): Path to the dataset root
    
    Returns:
    tuple: (features, labels)
    """
    features, labels = [], []
    file_paths = find_audio_files(directory)

    for file_path in file_paths:
        try:
            print(f"[INFO] Processing file: {file_path}")

            # Extract emotion label from filename
            filename = os.path.basename(file_path)
            emotion_code = int(filename.split('-')[2])  # Extract emotion label
            emotion_label = emotion_code - 1  # Convert to zero-based index

            # Load audio
            y, sr = load_audio(file_path)
            if y is not None and sr is not None:
                feature_vector = extract_features(y, sr)
                if feature_vector is not None and len(feature_vector) > 0:
                    features.append(feature_vector)
                    labels.append(emotion_label)
                else:
                    print(f"[WARNING] No features extracted for {file_path}")
            else:
                print(f"[ERROR] Failed to load {file_path}")

        except Exception as e:
            print(f"[ERROR] Error processing {file_path}: {e}")
            continue

    features_array = np.array(features, dtype=np.float32)
    labels_array = np.array(labels, dtype=np.int32)

    print(f"[INFO] Extracted features shape: {features_array.shape}")

    return features_array, labels_array

# ========================
# 🚀 RUNNING FEATURE EXTRACTION
# ========================
if __name__ == "__main__":
    dataset_path = "data/raw"  # Change this if needed
    print(f"[INFO] Extracting features from {dataset_path}...")

    features, labels = extract_features_batch(dataset_path)
    print(f"[INFO] Feature extraction complete. Extracted {features.shape[0]} samples.")
