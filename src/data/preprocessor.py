import numpy as np
from typing import Tuple
import librosa

class AudioPreprocessor:
    """Class for preprocessing audio data with normalization and resampling capabilities."""
    
    def __init__(self, target_sample_rate: int = 16000):
        """
        Initialize the preprocessor with target sample rate.
        
        Args:
            target_sample_rate (int): Target sampling rate in Hz (default: 16000)
        """
        self.target_sample_rate = target_sample_rate

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to have zero mean and unit variance.
        
        Args:
            audio (np.ndarray): Input audio signal
            
        Returns:
            np.ndarray: Normalized audio signal
        """
        audio_mean = audio.mean()
        audio_std = audio.std()
        if audio_std > 0:
            return (audio - audio_mean) / audio_std
        return audio
    
    def resample_audio(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate if necessary.
        
        Args:
            audio (np.ndarray): Input audio signal
            orig_sr (int): Original sampling rate
            
        Returns:
            np.ndarray: Resampled audio signal
        """
        if orig_sr != self.target_sample_rate:
            return librosa.resample(
                y=audio, 
                orig_sr=orig_sr, 
                target_sr=self.target_sample_rate
            )
        return audio

    def preprocess(self, audio: np.ndarray, orig_sr: int) -> Tuple[np.ndarray, int]:
        """
        Apply full preprocessing pipeline: resampling and normalization.
        
        Args:
            audio (np.ndarray): Input audio signal
            orig_sr (int): Original sampling rate
            
        Returns:
            Tuple[np.ndarray, int]: Preprocessed audio and new sample rate
        """
        # First resample
        resampled_audio = self.resample_audio(audio, orig_sr)
        
        # Then normalize
        normalized_audio = self.normalize_audio(resampled_audio)
        
        return normalized_audio, self.target_sample_rate