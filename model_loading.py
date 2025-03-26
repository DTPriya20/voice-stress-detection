import joblib
import numpy as np

# Load the trained model and preprocessing files
model_path = "models/emotion_classifier.pkl"
scaler_path = "models/scaler.pkl"
feature_selector_path = "models/feature_selector.pkl"
selected_features_path = "models/selected_feature_indices.pkl"

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_selector = joblib.load(feature_selector_path)
    selected_features = joblib.load(selected_features_path)

    print("✅ Model and preprocessing files loaded successfully.")

    # Generate a random test sample (Replace with actual features later)
    random_sample = np.random.rand(34)  # Ensure this matches feature count before selection
    
    # Preprocess: scale and select features
    random_sample_scaled = scaler.transform([random_sample])
    random_sample_selected = feature_selector.transform(random_sample_scaled)

    # Make prediction
    prediction = model.predict(random_sample_selected)

    print(f"🎯 Predicted emotion label: {prediction[0]}")

except Exception as e:
    print(f"❌ Error during prediction: {e}")
