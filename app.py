import streamlit as st
import librosa
import numpy as np
import os
import joblib  # ✅ Use `joblib` instead of `pickle`
import soundfile as sf
from datetime import datetime

# ✅ Ensure this is the first Streamlit command
st.set_page_config(page_title="Emotion & Stress Detector", page_icon="🎤", layout="wide")

# --- Load models ---
@st.cache_resource
def load_models():
    """Loads the classifier, feature selector, scaler, and selected feature indices."""
    try:
        model_dir = "models"

        classifier = joblib.load(os.path.join(model_dir, "emotion_classifier.pkl"))  # ✅ FIXED
        feature_selector = joblib.load(os.path.join(model_dir, "feature_selector.pkl"))  # ✅ FIXED
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))  # ✅ FIXED
        selected_features = joblib.load(os.path.join(model_dir, "selected_feature_indices.pkl"))  # ✅ FIXED

        return classifier, feature_selector, scaler, selected_features
    except Exception as e:
        st.error(f"🚨 Model loading error: {e}")
        return None, None, None, None

classifier, feature_selector, scaler, selected_features = load_models()

# --- Utility function for saving uploaded file temporarily ---
def create_temp_file(uploaded_file):
    """Save uploaded file to a temporary location."""
    try:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        return temp_path
    except Exception as e:
        st.error(f"⚠️ Failed to save file: {e}")
        return None

# --- Feature Extraction ---
def extract_features(audio_data, sample_rate):
    """Extracts MFCCs and applies feature selection/scaling."""
    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs, axis=1)

        # ✅ Ensure selected features match the trained model
        if selected_features is None:
            st.error("🚨 Feature selector is missing!")
            return None
        
        # ✅ Validate indices within range
        valid_indices = [i for i in selected_features if i < len(mfccs_mean)]
        if len(valid_indices) != len(selected_features):
            st.warning("⚠️ Some selected feature indices were out of range and have been ignored.")

        # ✅ Ensure extracted feature count matches scaler expectation
        if len(valid_indices) != scaler.n_features_in_:
            st.error(f"❌ Feature count mismatch! Model expects {scaler.n_features_in_}, but got {len(valid_indices)}.")
            return None

        mfccs_selected = mfccs_mean[valid_indices]
        mfccs_scaled = scaler.transform([mfccs_selected])

        return mfccs_scaled
    except Exception as e:
        st.error(f"❌ Feature extraction failed: {e}")
        return None



# --- Audio Analysis & Prediction ---
def analyze_audio(audio_data=None, sample_rate=None, uploaded_file=None):
    """Processes and predicts emotion/stress levels from audio."""
    progress_bar = st.progress(10)

    if classifier is None or feature_selector is None or scaler is None:
        st.error("🚨 Missing models! Ensure all model files are available in the 'models' folder.")
        return False

    progress_bar.progress(30)

    try:
        if uploaded_file:
            temp_path = create_temp_file(uploaded_file)
            if temp_path:
                y, sr = librosa.load(temp_path, sr=None)
                os.unlink(temp_path)  # ✅ Remove temp file after processing
            else:
                st.error("❌ Failed to process uploaded file.")
                return False
        elif audio_data is not None and sample_rate is not None:
            y, sr = audio_data, sample_rate
        else:
            st.error("⚠️ No valid audio input.")
            return False

        progress_bar.progress(60)

        features = extract_features(y, sr)
        if features is None:
            return False

        progress_bar.progress(80)

        prediction_probs = classifier.predict_proba(features)[0]
        predicted_class = np.argmax(prediction_probs)
        class_labels = classifier.classes_
        emotion = class_labels[predicted_class]
        stress_level = "High" if emotion in ["Angry", "Fear", "Sad"] else "Low"

        st.session_state.update({
            "prediction": predicted_class,
            "emotion": emotion,
            "stress_level": stress_level,
            "proba": prediction_probs.tolist(),
            "emotion_labels": class_labels.tolist(),
            "audio_data": y,
            "sample_rate": sr,
            "analyzed": True,
            "analysis_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        progress_bar.progress(100)
        return True
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        return False

# --- Streamlit UI with Tabs ---
st.title("🎤 Emotion & Stress Level Detector")

tabs = st.tabs(["🎵 Upload Audio", "🔍 Analysis", "📊 Prediction Results", "ℹ️ About"])

# --- Upload Audio Tab ---
with tabs[0]:
    st.header("🎵 Upload or Record Audio")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    
    if uploaded_file:
        st.success("✅ Audio file uploaded successfully!")

    if st.button("Analyze Audio"):
        if uploaded_file:
            analyze_audio(uploaded_file=uploaded_file)
        else:
            st.error("⚠️ Please upload an audio file first.")

# --- Analysis Tab ---
with tabs[1]:
    st.header("🔍 Audio Analysis")
    
    if "analyzed" in st.session_state and st.session_state.analyzed:
        st.write(f"📅 **Analysis Time:** {st.session_state.analysis_time}")
        st.write("✅ Audio processed successfully!")
    else:
        st.warning("⚠️ No analysis available yet. Upload an audio file in the **Upload Audio** tab.")

# --- Prediction Results Tab ---
with tabs[2]:
    st.header("📊 Emotion Prediction Results")
    
    if "analyzed" in st.session_state and st.session_state.analyzed:
        st.write(f"**Predicted Emotion:** {st.session_state.emotion}")
        st.write(f"**Stress Level:** {st.session_state.stress_level}")

        # Probability distribution
        st.subheader("📊 Prediction Probabilities")
        probabilities = st.session_state.proba
        labels = st.session_state.emotion_labels
        st.bar_chart({label: prob for label, prob in zip(labels, probabilities)})
    else:
        st.warning("⚠️ No predictions yet. Upload an audio file in the **Upload Audio** tab.")

# --- About Tab ---
with tabs[3]:
    st.header("ℹ️ About This App")
    st.markdown("""
    This application uses **Machine Learning** to detect emotions from audio recordings.  
    - 🎵 **Upload an audio file** or record live input.  
    - 🧠 **Analyze the speech** and extract meaningful features.  
    - 📊 **Predict emotions & stress levels** using trained AI models.  
    - Built using **Streamlit, Librosa, and Scikit-learn**.  
    """)
