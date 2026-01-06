"""SeamlessM4T model wrapper for ASR evaluation.

Facebook's SeamlessM4T is a multilingual multimodal model supporting
speech-to-text, text-to-speech, and speech-to-speech translation.
This wrapper exposes only the speech-to-text (ASR) functionality.

Model: facebook/seamless-m4t-v2-large
- ~2.3B parameters
- Supports 100+ languages including German
- Requires GPU + half precision for practical inference
"""

import torch
import torchaudio
from pathlib import Path
from typing import Optional, Dict

from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText


# Language code mapping for SeamlessM4T
# SeamlessM4T uses specific language codes that may differ from ISO codes
LANGUAGE_CODE_MAP = {
    "de": "deu",      # German (ISO 639-1 -> SeamlessM4T)
    "deu": "deu",     # German (ISO 639-3)
    "german": "deu",  # Natural language name
    "gsw": "deu",     # Swiss German -> German (no Swiss German in model)
}


class SeamlessM4TModel:
    """SeamlessM4T ASR model wrapper for Swiss German evaluation."""

    def __init__(
        self,
        model_name: str = "facebook/seamless-m4t-v2-large",
        device: str = None,
        use_half_precision: bool = True,
    ):
        """
        Initialize SeamlessM4T model for ASR.

        Args:
            model_name: Hugging Face model name (default: facebook/seamless-m4t-v2-large)
            device: Device to run on (auto-detected if None)
            use_half_precision: Use float16 for reduced memory (recommended for GPU)
        """
        self.model_name = model_name
        self.use_half_precision = use_half_precision

        # Device selection (consistent with other model wrappers)
        self.device = device if device else (
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        # Determine dtype based on device and user preference
        if self.use_half_precision and self.device == "cuda":
            self.dtype = torch.float16
        elif self.use_half_precision and self.device == "mps":
            # MPS supports float16 but can be unstable; use float32 for safety
            self.dtype = torch.float32
        else:
            self.dtype = torch.float32

        print(f"Loading SeamlessM4T model '{self.model_name}' on {self.device} ({self.dtype})...")

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully.")

    def _resolve_language_code(self, language: str) -> str:
        """
        Resolve user-provided language code to SeamlessM4T format.

        Args:
            language: User-provided language code (e.g., 'de', 'deu', 'german')

        Returns:
            SeamlessM4T-compatible language code
        """
        lang_lower = language.lower()
        if lang_lower in LANGUAGE_CODE_MAP:
            return LANGUAGE_CODE_MAP[lang_lower]
        # If not in map, return as-is (let the model handle it)
        return lang_lower

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = "de",
    ) -> Dict[str, str]:
        """
        Transcribe audio file to text using SeamlessM4T.

        Args:
            audio_path: Path to audio file (FLAC, WAV, MP3, etc.)
            language: Target language code (default: 'de' for German)
                     SeamlessM4T uses this for ASR output language.

        Returns:
            Dictionary with 'text' key containing transcription
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio using torchaudio (consistent with other wrappers)
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
        except Exception as e:
            raise ValueError(f"Failed to load audio: {e}") from e

        if waveform.numel() == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")

        # Resample to 16kHz if needed (SeamlessM4T expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resolve language code
        tgt_lang = self._resolve_language_code(language) if language else "deu"

        # Process audio through SeamlessM4T processor
        # Processor expects numpy array or list
        audio_inputs = self.processor(
            audios=waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )

        # Move inputs to device
        audio_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in audio_inputs.items()
        }

        # Generate transcription
        with torch.no_grad():
            output_tokens = self.model.generate(
                **audio_inputs,
                tgt_lang=tgt_lang,
            )

        # Decode output tokens to text
        transcription = self.processor.decode(
            output_tokens[0].tolist(),
            skip_special_tokens=True,
        )

        return {"text": transcription.strip()}
