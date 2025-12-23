"""
IMPROVED evaluator.py with SSH-keepalive-friendly progress logging

Key changes:
1. Added tqdm progress bar for visual feedback
2. Added periodic print statements every N samples (keeps SSH alive)
3. Added elapsed time tracking
4. Added ETA estimation
5. All changes are in evaluate_dataset() method only

This prevents SSH timeout during long evaluations by ensuring regular stdout activity.
"""

import whisper
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta
from tqdm import tqdm  
from transformers import WhisperProcessor, WhisperForConditionalGeneration 
from src.evaluation import metrics
from src.models.wav2vec2_model import Wav2Vec2Model
from src.models.mms_model import MMSModel
from src.config import FHNW_SWISS_GERMAN_ROOT

class ASREvaluator:
    def __init__(self, model_type: str = "whisper", model_name: str = "base", device: str = None, lm_path: str = None):
        """
        Initialize the ASR Evaluator.
        
        Args:
            model_type: 'whisper', 'whisper-hf', 'wav2vec2', or 'mms'
            model_name: Model identifier
            device: Device string
            lm_path: Optional path to KenLM file
        """
        if model_type not in ["whisper", "whisper-hf", "wav2vec2", "mms"]:
            raise ValueError(f"model_type must be 'whisper', 'whisper-hf', 'wav2vec2', or 'mms', got: {model_type}")

        self.model_type = model_type
        self.model_name = model_name
        self.lm_path = lm_path
        self.device = device if device else (
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.model = None
        self.processor = None  # ← used by wav2vec2/mms/whisper-hf if needed

    def load_model(self):
        """Load the appropriate model."""
        if self.model_type == "whisper":
            print(f"Loading Whisper model '{self.model_name}' on {self.device}...")
            try:
                self.model = whisper.load_model(self.model_name, device=self.device)
                print("Model loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load Whisper model: {e}") from e

        elif self.model_type == "whisper-hf":
            print(f"Loading Hugging Face Whisper model '{self.model_name}' on {self.device}...")
            try:
                self.processor = WhisperProcessor.from_pretrained(self.model_name)
                self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                print("HF Whisper model loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load HF Whisper model: {e}") from e

        elif self.model_type == "wav2vec2":
            try:
                self.model = Wav2Vec2Model(
                    model_name=self.model_name,
                    device=self.device,
                    lm_path=self.lm_path
                )
                # Note: NO self.model.load_model() - method doesn't exist
                # Model is fully loaded after __init__() completes
            except Exception as e:
                raise RuntimeError(f"Failed to load Wav2Vec2 model: {e}") from e

        elif self.model_type == "mms":
            try:
                self.model = MMSModel(
                    model_name=self.model_name,
                    device=self.device,
                    lm_path=self.lm_path
                )
                # Note: NO self.model.load_model() - method doesn't exist
                # Model is fully loaded after __init__() completes
            except Exception as e:
                raise RuntimeError(f"Failed to load MMS model: {e}") from e

    def _get_transcription(self, audio_path: Path) -> str:
        """Get transcription from loaded model."""
        if self.model_type == "whisper":
            audio = whisper.load_audio(str(audio_path))
            # ✅ DETERMINISTIC WHISPER PARAMETERS
            result = self.model.transcribe(
                audio, 
                language="de",
                temperature=0.0,      # Deterministic decoding
                beam_size=5,          # Consistent beam search
                best_of=5,            # Deterministic candidate selection
                fp16=False            # UNCONDITIONAL FP32 for reproducibility
            )
            return result['text']

        elif self.model_type == "whisper-hf":
            if self.model is None or self.processor is None:
                raise RuntimeError("HF Whisper model not loaded. Call load_model() first.")

            # reuse whisper's audio loader → 16 kHz mono float32 numpy array
            audio = whisper.load_audio(str(audio_path))

            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            )
            input_features = inputs.input_features.to(self.device)

            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    language="de",
                    task="transcribe",
                )

            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()
            return transcription

        elif self.model_type in ["wav2vec2", "mms"]:
            result = self.model.transcribe(audio_path)
            if isinstance(result, dict) and "text" in result:
                return result["text"]
            # Fallback for unexpected return type (shouldn't happen)
            return str(result)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def evaluate_dataset(
        self,
        metadata_path: str,
        audio_base_path: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Dict:
        """
        Evaluate model on Swiss German dataset with SSH-keepalive-friendly logging.
        
        IMPROVED: Now includes progress bar and periodic status updates to prevent
        SSH timeout during long evaluations.
        
        Args:
            metadata_path: Path to test.tsv file
            audio_base_path: Base path for audio files
            limit: Optional limit on number of samples
            
        Returns:
            Dictionary with evaluation results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Use config path if not specified
        if audio_base_path is None:
            audio_base_path = FHNW_SWISS_GERMAN_ROOT / "clips"
        else:
            audio_base_path = Path(audio_base_path)

        try:
            df = pd.read_csv(metadata_path, sep='\t')
        except Exception as e:
            raise ValueError(f"Failed to read metadata file: {e}") from e

        required_columns = {'path', 'sentence', 'accent'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Metadata file must contain columns: {required_columns}")

        if limit is not None and limit > 0:
            df = df.head(limit)

        results = []
        failed_samples = 0
        total_samples = len(df)

        # Print header with timestamp
        print(f"\n{'='*60}")
        print(f"Processing {total_samples} samples...")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        # Track start time for ETA calculation
        start_time = datetime.now()

        # Use tqdm for progress bar
        # This provides visual feedback AND regular stdout updates
        progress_bar = tqdm(
            df.iterrows(),
            total=total_samples,
            desc="Evaluating",
            unit="sample",
            ncols=100,  # Fixed width for better formatting
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        for idx, row in progress_bar:
            reference = row['sentence']
            accent = row['accent']
            
            # Audio path resolution logic (unchanged)
            if 'audio_path' in df.columns and pd.notna(row.get('audio_path')) and row.get('audio_path'):
                audio_path = Path(row['audio_path']).resolve()
                try:
                    audio_path.relative_to(audio_base_path.resolve())
                except ValueError:
                    failed_samples += 1
                    continue
            else:
                audio_filename = Path(row['path']).name
                audio_path = (audio_base_path / audio_filename).resolve()
                try:
                    audio_path.relative_to(audio_base_path.resolve())
                except ValueError:
                    failed_samples += 1
                    continue

            if not audio_path.exists():
                failed_samples += 1
                continue

            try:
                hypothesis = self._get_transcription(audio_path)
            except Exception as e:
                print(f"\n⚠️  Transcription failed for {audio_path.name}: {e}")
                failed_samples += 1
                continue

            wer = metrics.calculate_wer(reference, hypothesis)
            cer = metrics.calculate_cer(reference, hypothesis)
            bleu = metrics.calculate_bleu_score(reference, hypothesis)

            results.append({
                'audio_file': audio_path.name,
                'dialect': accent,
                'reference': reference,
                'hypothesis': hypothesis,
                'wer': wer,
                'cer': cer,
                'bleu': bleu
            })

            # ✅ IMPROVEMENT 4: Print progress milestone every 10 samples
            # This ensures stdout activity at least every ~50-100 seconds
            # (assuming ~5-10 seconds per sample)
            samples_processed = len(results)
            if samples_processed % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time_per_sample = elapsed / samples_processed
                remaining_samples = total_samples - samples_processed
                eta_seconds = avg_time_per_sample * remaining_samples
                eta = datetime.now() + timedelta(seconds=eta_seconds)
                
                print(f"Processed {samples_processed}/{total_samples} samples...")
                print(f"  Average: {avg_time_per_sample:.1f}s/sample")
                print(f"  ETA: {eta.strftime('%H:%M:%S')}")

        # Close progress bar
        progress_bar.close()

        # ✅ IMPROVEMENT 5: Print completion summary with timing
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*60}")
        print(f"✅ Evaluation complete!")
        print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {timedelta(seconds=int(total_duration))}")
        if results:
            print(f"Average: {total_duration/len(results):.2f}s/sample")
        else:
            print("No successful transcriptions")
        print(f"{'='*60}\n")

        # If no results, return early with zeroed metrics
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

        # Calculate aggregate metrics (unchanged)
        overall_wer = sum(r['wer'] for r in results) / len(results)
        
        references = [r['reference'] for r in results]
        hypotheses = [r['hypothesis'] for r in results]
        cer_result = metrics.batch_cer(references, hypotheses)
        overall_cer = cer_result['overall_cer']
        
        bleu_result = metrics.batch_bleu(references, hypotheses)
        overall_bleu = bleu_result['overall_bleu']

        # Aggregate per-dialect metrics (unchanged)
        dialects = set(r['dialect'] for r in results)
        per_dialect_wer = {}
        per_dialect_cer = {}
        per_dialect_bleu = {}

        for dialect in dialects:
            dialect_samples = [r for r in results if r['dialect'] == dialect]
            if dialect_samples:
                per_dialect_wer[dialect] = sum(r['wer'] for r in dialect_samples) / len(dialect_samples)
                refs = [r['reference'] for r in dialect_samples]
                hyps = [r['hypothesis'] for r in dialect_samples]
                per_dialect_cer[dialect] = metrics.batch_cer(refs, hyps)['overall_cer']
                per_dialect_bleu[dialect] = metrics.batch_bleu(refs, hyps)['overall_bleu']

        return {
            'overall_wer': overall_wer,
            'overall_cer': overall_cer,
            'overall_bleu': overall_bleu,
            'per_dialect_wer': per_dialect_wer,
            'per_dialect_cer': per_dialect_cer,
            'per_dialect_bleu': per_dialect_bleu,
            'total_samples': len(results),
            'failed_samples': failed_samples,
            'samples': results
        }