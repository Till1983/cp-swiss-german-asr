import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pathlib import Path
from typing import Optional, Dict
import torchaudio
import numpy as np

# ✅ Try importing decoder
try:
    from pyctcdecode import build_ctcdecoder
    _HAS_PYCTCDECODE = True
except ImportError:
    _HAS_PYCTCDECODE = False

class Wav2Vec2Model:
    def __init__(self, model_name: str = "facebook/wav2vec2-large-xlsr-53-german", device: str = None, lm_path: str = None):
        """
        Initialize Wav2Vec2 model for ASR.
        
        Args:
            model_name: Hugging Face model name
            device: Device to run on
            lm_path: Optional path to KenLM .arpa or .bin file
        """
        self.model_name = model_name
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() 
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.lm_path = lm_path
        self.decoder = None
        
        print(f"Loading Wav2Vec2 model '{self.model_name}' on {self.device}...")
        
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        except (TypeError, OSError) as e:
            raise ValueError(f"Failed to load processor: {str(e)}") from e
        
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            use_safetensors=True
        )
        self.model.to(self.device)
        self.model.eval()
        
        # ✅ Initialize Decoder if LM provided
        if self.lm_path:
            if not _HAS_PYCTCDECODE:
                print("⚠️ Warning: lm_path provided but 'pyctcdecode' not installed. Falling back to greedy search.")
            elif not Path(self.lm_path).exists():
                print(f"⚠️ Warning: LM file not found at {self.lm_path}. Falling back to greedy search.")
            else:
                print(f"Initializing Beam Search Decoder with {self.lm_path}...")
                self._init_decoder()

        print("Model loaded successfully.")

    def _init_decoder(self):
        """Builds the pyctcdecode Beam Search Decoder."""
        # Get vocab from tokenizer
        vocab = self.processor.tokenizer.get_vocab()
        # Sort vocab by index to get character list
        sorted_vocab = [k.lower() for k, v in sorted(vocab.items(), key=lambda item: item[1])]
        
        try:
            self.decoder = build_ctcdecoder(
                labels=sorted_vocab,
                kenlm_model_path=str(self.lm_path),
            )
            print("✅ Beam Search Decoder initialized.")
        except Exception as e:
            print(f"❌ Failed to initialize decoder: {e}")
            self.decoder = None

    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> Dict[str, str]:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
        except Exception as e:
            raise ValueError(f"Failed to load audio: {e}") from e
        
        if waveform.numel() == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        audio_input = self.processor(
            waveform.squeeze().numpy(), 
            return_tensors="pt", 
            sampling_rate=16000
        )
        
        audio_input = {k: v.to(self.device) for k, v in audio_input.items()}
        
        with torch.no_grad():
            logits = self.model(**audio_input).logits
        
        # ✅ DECODING STRATEGY
        if self.decoder:
            # Beam Search with LM
            logits_np = logits.squeeze().cpu().numpy()
            transcription = self.decoder.decode(logits_np)
        else:
            # Greedy Search
            predicted_ids = logits.argmax(dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return {"text": transcription}