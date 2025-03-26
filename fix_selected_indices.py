import joblib
import numpy as np

# Load the incorrect feature selection file
selected_indices_path = "models/selected_feature_indices.pkl"

try:
    selected_indices = joblib.load(selected_indices_path)
    print(f"🔍 Original Selected Indices: {selected_indices}")

    # ✅ Remove out-of-range indices
    max_feature_index = 39  # Since extracted features range from 0 to 39
    valid_indices = np.array([idx for idx in selected_indices if idx <= max_feature_index])

    print(f"✅ Fixed Selected Indices: {valid_indices}")

    # Save the corrected feature selection file
    joblib.dump(valid_indices, selected_indices_path)
    print("✅ Corrected selected_feature_indices.pkl saved successfully!")

except Exception as e:
    print(f"❌ Error: {e}")
