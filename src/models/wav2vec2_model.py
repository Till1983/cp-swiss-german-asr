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
    def __init__(self, model_name: str = "aware-ai/wav2vec2-large-xlsr-53-german-with-lm", device: str = None, lm_path: str = None):
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
        self._can_set_tokenizer_lang = False
        self._has_model_adapter = False
        self._lang_warning_printed = False
        self._adapter_warning_printed = False
        
        print(f"Loading Wav2Vec2 model '{self.model_name}' on {self.device}...")

        # ✅ Check if model path exists locally (before try block)
        is_local_path = Path(model_name).exists()

        try:
            if is_local_path:
                print(f"   Detected local model path, using local_files_only=True")
                self.processor = Wav2Vec2Processor.from_pretrained(
                    model_name, 
                    local_files_only=True
                )
            else:
                print(f"   Loading from HuggingFace Hub")
                self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        except (TypeError, OSError) as e:
            raise ValueError(f"Failed to load processor: {str(e)}") from e

        # Load model with same logic
        if is_local_path:
            self.model = Wav2Vec2ForCTC.from_pretrained(
                model_name, 
                use_safetensors=True, 
                local_files_only=True
            )
        else:
            self.model = Wav2Vec2ForCTC.from_pretrained(
                model_name, 
                use_safetensors=True
            )
        
        self.model.to(self.device)
        self.model.eval()

        # Detect language capabilities at initialisation to avoid repeated checks during transcription
        self._can_set_tokenizer_lang = self._detect_language_capability()
        self._has_model_adapter = hasattr(self.model, "load_adapter")
        
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
        vocab = self.processor.tokenizer.get_vocab()
        # ✅ DON'T lowercase - keep as-is from vocab
        sorted_vocab = [k for k, v in sorted(vocab.items(), key=lambda item: item[1])]
        
        try:
            self.decoder = build_ctcdecoder(
                labels=sorted_vocab,
                kenlm_model_path=str(self.lm_path),
            )
            print("✅ Beam Search Decoder initialized.")
        except Exception as e:
            print(f"❌ Failed to initialize decoder: {e}")
            print(f"⚠️ Falling back to greedy decoding (LM will not be used)")
            self.decoder = None  # ✅ SAFE: Fall back gracefully

    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> Dict[str, str]:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language code for multilingual models (e.g., 'de' for German).
                     - Required for multilingual Wav2Vec2 models with language adapters
                     - Ignored by monolingual German-specific models
                     - When provided, attempts to load language-specific adapter

        Returns:
            Dictionary with 'text' key containing transcription
            
        Note:
            For multilingual models (e.g., voidful/wav2vec2-xlsr-multilingual-56),
            the language parameter enables proper adapter switching. For monolingual
            German models, this parameter is accepted but safely ignored.
        """
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
        
        # ✅ Language handling
        # Only attempt target-lang switching when the tokenizer exposes a language map
        if language:
            if self._can_set_tokenizer_lang:
                try:
                    self.processor.tokenizer.set_target_lang(language)
                except Exception as e:
                    if not self._lang_warning_printed:
                        print(f"ℹ️ Could not set tokenizer language to '{language}': {e}")
                        self._lang_warning_printed = True
            else:
                if not self._lang_warning_printed:
                    print("ℹ️ Language parameter ignored: tokenizer has no language mapping (monolingual CTC vocab).")
                    self._lang_warning_printed = True

            if self._has_model_adapter:
                try:
                    self.model.load_adapter(language)
                except Exception as e:
                    if not self._adapter_warning_printed:
                        print(f"ℹ️ Could not load adapter for language '{language}': {e}")
                        print("   Continuing with default model configuration (likely monolingual model)")
                        self._adapter_warning_printed = True
        
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

    def _detect_language_capability(self) -> bool:
        """Check whether tokenizer supports multilingual target language switching."""
        tokenizer = getattr(self.processor, "tokenizer", None)
        if not tokenizer or not hasattr(tokenizer, "set_target_lang"):
            return False

        # Some multilingual tokenizers expose lang2id or languages collections
        lang_map = getattr(tokenizer, "lang2id", None) or getattr(tokenizer, "languages", None)
        return bool(lang_map)