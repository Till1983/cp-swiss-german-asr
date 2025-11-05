import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pathlib import Path
from typing import Optional, Dict
import torchaudio

class Wav2Vec2Model:
    def __init__(self, model_name: str = "facebook/wav2vec2-large-xlsr-53-german", device: str = None):
        """
        Initialize Wav2Vec2 model for ASR.
        
        Args:
            model_name: Hugging Face model name
            device: Device to run on ("cuda", "mps", or "cpu"). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() 
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        
        print(f"Loading Wav2Vec2 model '{self.model_name}' on {self.device}...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            use_safetensors=True  # Prefer SafeTensors format to avoid double download
        )
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> Dict[str, str]:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language parameter (not used for Wav2Vec2, kept for compatibility)
            
        Returns:
            Dictionary with 'text' key containing transcription
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio file is invalid or empty
        """
        # Convert to Path object if string
        audio_path = Path(audio_path)
        
        # Check if file exists
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Check if it's a file (not a directory)
        if not audio_path.is_file():
            raise ValueError(f"Path is not a file: {audio_path}")
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(str(audio_path))
        except Exception as e:
            raise ValueError(f"Failed to load audio file {audio_path}: {str(e)}")
        
        # Check if audio is empty
        if waveform.numel() == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Process audio
        audio_input = self.processor(
            waveform.squeeze().numpy(), 
            return_tensors="pt", 
            sampling_rate=16000
        )
        
        # Move input to device
        audio_input = {k: v.to(self.device) for k, v in audio_input.items()}
        
        # Transcribe
        with torch.no_grad():
            logits = self.model(**audio_input).logits
        
        predicted_ids = logits.argmax(dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return {"text": transcription}