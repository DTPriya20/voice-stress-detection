import os
import glob
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from feature_extraction import extract_features_batch

def create_directories():
    """Create necessary directories for the project"""
    directories = ['models', 'data', 'data/raw', 'data/processed']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

def find_audio_files():
    """Find all audio files inside the dataset directory."""
    dataset_path = 'data/raw'
    if not os.path.exists(dataset_path):
        print("Dataset directory not found.")
        return []
    
    audio_files = glob.glob(os.path.join(dataset_path, '**', '*.wav'), recursive=True)
    if not audio_files:
        print("No audio files found. Please check the dataset.")
        return []
    
    print(f"Found {len(audio_files)} audio files.")
    return audio_files

def train_model(X_train, X_test, y_train, y_test):
    """Train and evaluate a RandomForestClassifier model."""
    print(f"Feature shape before training: {X_train.shape[1]} features")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Feature Selection - Keep the top 20 features
    selector = SelectKBest(f_classif, k=20)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    print(f"Feature shape after selection: {X_train_selected.shape[1]} features")

    # Train a Random Forest model
    model = RandomForestClassifier(
        n_estimators=500,  
        max_depth=30,  
        min_samples_split=4,  
        bootstrap=False,  
        random_state=42
    )

    model.fit(X_train_selected, y_train)

    # Save feature selector & indices for prediction
    joblib.dump(selector, "models/feature_selector.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    
    selected_indices = selector.get_support(indices=True)
    joblib.dump(selected_indices, "models/selected_feature_indices.pkl")

    return model, scaler

if __name__ == "__main__":
    create_directories()

    # Use the directory for feature extraction
    dataset_path = 'data/raw'
    if not os.path.exists(dataset_path):
        print(f"Dataset directory '{dataset_path}' not found. Exiting.")
        exit()

    # Extract features from the audio files in the directory
    print(f"Extracting features from {dataset_path}...")
    X, y = extract_features_batch(dataset_path)

    if X.size == 0:
        print("Feature extraction failed. Exiting.")
        exit()

    print(f"Total features extracted: {X.shape[1]}")

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the model
    model, scaler = train_model(X_train, X_test, y_train, y_test)

    # Save the trained model
    joblib.dump(model, "models/emotion_classifier.pkl")

    print("\n✅ Model training complete. Saved model to 'models/emotion_classifier.pkl'.")
