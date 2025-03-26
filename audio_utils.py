import librosa
import numpy as np
import soundfile as sf
import tempfile
import os

def load_audio(file_path, duration=3.0, offset=0.5, sr=22050):
    """
    Load an audio file using librosa
    
    Parameters:
    file_path (str): Path to the audio file
    duration (float): Duration of audio to load (in seconds)
    offset (float): Start reading after this time (in seconds)
    sr (int): Target sampling rate
    
    Returns:
    tuple: (audio_data, sample_rate)
    """
    try:
        y, sr = librosa.load(file_path, duration=duration, offset=offset, sr=sr)
        return y, sr
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None

def save_audio_from_stream(audio_frames, output_path, sample_rate=48000):
    """
    Save audio frames from WebRTC stream to a file
    
    Parameters:
    audio_frames (list): List of audio frames
    output_path (str): Path to save the audio file
    sample_rate (int): Sampling rate of the audio
    
    Returns:
    bool: Success status
    """
    try:
        # Concatenate all audio frames
        if len(audio_frames) > 0:
            audio_data = np.concatenate(audio_frames, axis=0)
            
            # Normalize audio
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Save as WAV file
            sf.write(output_path, audio_data, sample_rate)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error saving audio: {e}")
        return False

def create_temp_file(uploaded_file):
    """
    Create a temporary file from an uploaded file
    
    Parameters:
    uploaded_file: Streamlit UploadedFile object
    
    Returns:
    str: Path to temporary file or None if failed
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(uploaded_file.getvalue())
            return tmp.name
    except Exception as e:
        print(f"Error creating temporary file: {e}")
        return None