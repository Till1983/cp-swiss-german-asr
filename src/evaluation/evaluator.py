import whisper
import torch
import pandas as pd
from pathlib import Path
from typing import Dict
from src.evaluation import metrics
from src.models.wav2vec2_model import Wav2Vec2Model
from src.models.mms_model import MMSModel
from src.config import FHNW_SWISS_GERMAN_ROOT


class ASREvaluator:
    """
    ASR Evaluator for Whisper, Wav2Vec2, and MMS models on Swiss German audio datasets.
    """

    def __init__(self, model_type: str = "whisper", model_name: str = "base", device: str = None):
        """
        Initialize the ASR Evaluator.

        Args:
            model_type: Type of model ('whisper', 'wav2vec2', or 'mms')
            model_name: Model identifier
                       - Whisper: tiny/base/small/medium/large
                       - Wav2Vec2: HuggingFace model name
                       - MMS: HuggingFace model name
            device: Device to run on ("cuda", "mps", or "cpu"). Auto-detected if None.
        """
        if model_type not in ["whisper", "wav2vec2", "mms"]:
            raise ValueError(f"model_type must be 'whisper', 'wav2vec2', or 'mms', got: {model_type}")

        self.model_type = model_type
        self.model_name = model_name
        self.device = device if device else (
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.model = None

    def load_model(self):
        """
        Load the appropriate model onto the specified device.
        """
        if self.model_type == "whisper":
            print(f"Loading Whisper model '{self.model_name}' on {self.device}...")
            try:
                self.model = whisper.load_model(self.model_name, device=self.device)
                print("Model loaded successfully.")
            except Exception as e:
                raise ValueError(f"Failed to load Whisper model '{self.model_name}': {str(e)}")

        elif self.model_type == "wav2vec2":
            try:
                self.model = Wav2Vec2Model(model_name=self.model_name, device=self.device)
            except Exception as e:
                raise ValueError(f"Failed to load Wav2Vec2 model '{self.model_name}': {str(e)}") from e

        elif self.model_type == "mms":
            try:
                self.model = MMSModel(model_name=self.model_name, device=self.device)
            except Exception as e:
                raise ValueError(
                    f"Failed to load MMS model '{self.model_name}'. "
                    f"Ensure you're using an ASR model (e.g., facebook/mms-1b-all), "
                    f"not a TTS model (e.g., facebook/mms-tts). "
                    f"Error: {str(e)}"
                ) from e

    def _get_transcription(self, audio_path: Path) -> str:
        """
        Get transcription from the appropriate model.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcription text
        """
        if self.model_type == "whisper":
            audio = whisper.load_audio(str(audio_path))
            result = self.model.transcribe(audio, language="de")
            return result['text']

        elif self.model_type == "wav2vec2":
            result = self.model.transcribe(audio_path, language="de")
            return result['text']

        elif self.model_type == "mms":
            # Use "deu" (ISO 639-3) for German
            result = self.model.transcribe(audio_path, language="deu")
            return result['text']

    def evaluate_dataset(self, metadata_path: str, audio_base_path: Path = None, limit: int = None) -> Dict:
        """
        Evaluate the model on a dataset defined by a TSV metadata file.

        Args:
            metadata_path: Path to TSV file with columns: path, sentence, accent
            audio_base_path: Base directory where audio files are stored.
                             If None, defaults to FHNW_SWISS_GERMAN_ROOT/clips from config.
            limit: Optional limit on number of samples to process

        Returns:
            Dictionary containing:
                - overall_wer: Overall WER across all samples
                - overall_cer: Overall CER across all samples
                - overall_bleu: Overall BLEU across all samples
                - per_dialect_wer: Dictionary of WER per dialect/accent
                - per_dialect_cer: Dictionary of CER per dialect/accent
                - per_dialect_bleu: Dictionary of BLEU per dialect/accent
                - total_samples: Total number of samples processed
                - failed_samples: Number of samples that failed
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Use config path if not specified
        if audio_base_path is None:
            audio_base_path = FHNW_SWISS_GERMAN_ROOT / "clips"
        else:
            audio_base_path = Path(audio_base_path)

        # Read metadata
        try:
            df = pd.read_csv(metadata_path, sep='\t')
        except Exception as e:
            raise ValueError(f"Failed to read metadata file: {e}") from e

        required_columns = {'path', 'sentence', 'accent'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Metadata file must contain columns: {required_columns}")

        # Apply limit if specified
        if limit is not None and limit > 0:
            df = df.head(limit)

        results = []
        failed_samples = 0

        print(f"Processing {len(df)} samples...")

        for idx, row in df.iterrows():
            reference = row['sentence']
            accent = row['accent']
            
            # PRIORITY 1: Use audio_path column if available (full absolute path)
            if 'audio_path' in df.columns and pd.notna(row.get('audio_path')) and row.get('audio_path'):
                audio_path = Path(row['audio_path']).resolve()
                # Validate the path is within expected boundaries
                try:
                    # This will raise ValueError if audio_path is not relative to audio_base_path
                    audio_path.relative_to(audio_base_path.resolve())
                except ValueError:
                    print(f"Warning: Audio path outside base directory: {audio_path}")
                    failed_samples += 1
                    continue
            else:
                # PRIORITY 2: Construct from base_path + filename
                audio_filename = Path(row['path']).name  # Extract just the filename
                audio_path = (audio_base_path / audio_filename).resolve()
                # Validate constructed path is within expected boundaries
                try:
                    audio_path.relative_to(audio_base_path.resolve())
                except ValueError:
                    print(f"Warning: Constructed audio path outside base directory: {audio_path}")
                    failed_samples += 1
                    continue

            try:
                # Check if audio file exists
                if not audio_path.exists():
                    print(f"Warning: Audio file not found: {audio_path}")
                    failed_samples += 1
                    continue

                # Get transcription using appropriate model
                hypothesis = self._get_transcription(audio_path)

                # Calculate metrics for this sample
                wer = metrics.calculate_wer(reference, hypothesis)
                cer = metrics.calculate_cer(reference, hypothesis)
                bleu = metrics.calculate_bleu_score(reference, hypothesis)

                results.append({
                    'audio_file': str(audio_path.name),
                    'dialect': accent,
                    'reference': reference,
                    'hypothesis': hypothesis,
                    'wer': wer,
                    'cer': cer,
                    'bleu': bleu
                })

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                failed_samples += 1
                continue

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(df)} samples...")

        # Calculate overall metrics
        if not results:
            return {
                'overall_wer': 0.0,
                'overall_cer': 0.0,
                'overall_bleu': 0.0,
                'per_dialect_wer': {},
                'per_dialect_cer': {},
                'per_dialect_bleu': {},
                'total_samples': 0,
                'failed_samples': failed_samples,
                'samples': []
            }

        # Calculate overall WER
        overall_wer = sum(r['wer'] for r in results) / len(results)

        # Calculate per-dialect WER
        per_dialect_wer = {}
        results_df = pd.DataFrame(results)
        for accent in results_df['dialect'].unique():
            accent_results = results_df[results_df['dialect'] == accent]
            per_dialect_wer[accent] = accent_results['wer'].mean()

        # Calculate overall CER
        references = [r['reference'] for r in results]
        hypotheses = [r['hypothesis'] for r in results]
        cer_result = metrics.batch_cer(references, hypotheses)
        overall_cer = cer_result['overall_cer']

        # Calculate overall BLEU
        bleu_result = metrics.batch_bleu(references, hypotheses)
        overall_bleu = bleu_result['overall_bleu']

        # Calculate per-dialect CER
        per_dialect_cer = {}
        for accent in results_df['dialect'].unique():
            accent_results = results_df[results_df['dialect'] == accent]
            accent_refs = accent_results['reference'].tolist()
            accent_hyps = accent_results['hypothesis'].tolist()
            accent_cer = metrics.batch_cer(accent_refs, accent_hyps)
            per_dialect_cer[accent] = accent_cer['overall_cer']

        # Calculate per-dialect BLEU
        per_dialect_bleu = {}
        for accent in results_df['dialect'].unique():
            accent_results = results_df[results_df['dialect'] == accent]
            accent_refs = accent_results['reference'].tolist()
            accent_hyps = accent_results['hypothesis'].tolist()
            accent_bleu = metrics.batch_bleu(accent_refs, accent_hyps)
            per_dialect_bleu[accent] = accent_bleu['overall_bleu']

        return {
            'overall_wer': overall_wer,
            'overall_cer': overall_cer,
            'overall_bleu': overall_bleu,
            'per_dialect_wer': per_dialect_wer,
            'per_dialect_cer': per_dialect_cer,
            'per_dialect_bleu': per_dialect_bleu,
            'total_samples': len(results),
            'failed_samples': failed_samples,
            'samples': results[:5]  # Return only first 5 samples for inspection
        }