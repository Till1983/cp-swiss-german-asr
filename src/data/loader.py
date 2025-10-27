import pandas as pd
import librosa
import numpy as np

def load_swiss_german_metadata(filepath: str) -> pd.DataFrame:
    """Load metadata from TSV file containing Swiss German speech data."""
    df = pd.read_csv(filepath, sep='\t')
    return df

def load_audio(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load and resample audio file to target sample rate."""
    try:
        audio, _ = librosa.load(audio_path, sr=sample_rate)
        return audio
    except Exception as e:
        raise RuntimeError(f"Error loading audio file {audio_path}: {str(e)}")