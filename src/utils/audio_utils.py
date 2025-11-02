import librosa
from pathlib import Path
from typing import Optional


def validate_audio_file(path: str) -> bool:
    """
    Check if audio file exists and has a valid extension.
    
    Args:
        path: Path to the audio file
        
    Returns:
        True if file exists and has valid extension, False otherwise
    """
    try:
        file_path = Path(path)
        valid_extensions = {'.flac', '.wav', '.mp3'}
        
        if not file_path.exists():
            return False
        
        if file_path.suffix.lower() not in valid_extensions:
            return False
        
        return True
    except Exception:
        return False


def get_audio_duration(path: str) -> Optional[float]:
    """
    Get duration of audio file in seconds.
    
    Args:
        path: Path to the audio file
        
    Returns:
        Duration in seconds, or None if error occurs
    """
    try:
        duration = librosa.get_duration(path=path)
        return duration
    except Exception:
        return None