import torch
from transformers import Wav2Vec2ForCTC, AutoProcessor
from pathlib import Path
from typing import Optional, Dict
import torchaudio
import numpy as np

try:
    from pyctcdecode import build_ctcdecoder
    _HAS_PYCTCDECODE = True
except ImportError:
    _HAS_PYCTCDECODE = False

class MMSModel:
    def __init__(self, model_name: str = "facebook/mms-1b-all", device: str = None, lm_path: str = None):
        """
        Initialize MMS model for ASR.
        """
        self.model_name = model_name
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() 
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.lm_path = lm_path
        self.decoder = None
        
        print(f"Loading MMS model '{self.model_name}' on {self.device}...")
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            use_safetensors=True
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize decoder only if LM is provided (Lazy init in transcribe to handle language switching)
        self.decoder = None
        
        # ✅ Initialize decoder immediately if LM provided
        if self.lm_path and _HAS_PYCTCDECODE:
            if Path(self.lm_path).exists():
                self._init_decoder()
            else:
                print(f"⚠️ Warning: LM file not found at {self.lm_path}")

        print("Model loaded successfully.")

    def _init_decoder(self):
        """Builds the pyctcdecode Beam Search Decoder."""
        if not _HAS_PYCTCDECODE:
            return
        
        print(f"Initializing KenLM decoder for MMS with {self.lm_path}...")
        vocab = self.processor.tokenizer.get_vocab()
        sorted_vocab = [k for k, v in sorted(vocab.items(), key=lambda item: item[1])]
        
        try:
            self.decoder = build_ctcdecoder(
                labels=sorted_vocab,
                kenlm_model_path=str(self.lm_path),
            )
            print("✅ MMS Beam Search Decoder initialized.")
        except Exception as e:
            print(f"❌ Failed to init decoder: {e}")
            print(f"⚠️ Falling back to greedy decoding (LM will not be used)")
            self.decoder = None  # ✅ SAFE: Fall back gracefully

    def transcribe(self, audio_path: Path, language: Optional[str] = "deu") -> Dict[str, str]:
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
        
        # Set target language
        if language:
            self.processor.tokenizer.set_target_lang(language)
            self.model.load_adapter(language)
            
            # Check if we need to init decoder (first run or re-init if needed)
            if self.lm_path and self.decoder is None:
                self._init_decoder()
        
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
            logits_np = logits.squeeze().cpu().numpy()
            transcription = self.decoder.decode(logits_np)
        else:
            predicted_ids = logits.argmax(dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return {"text": transcription}