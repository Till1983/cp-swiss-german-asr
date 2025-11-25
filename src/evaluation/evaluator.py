import whisper
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from src.evaluation import metrics
from src.models.wav2vec2_model import Wav2Vec2Model
from src.models.mms_model import MMSModel
from src.config import FHNW_SWISS_GERMAN_ROOT

class ASREvaluator:
    def __init__(self, model_type: str = "whisper", model_name: str = "base", device: str = None, lm_path: str = None):
        """
        Initialize the ASR Evaluator.
        
        Args:
            model_type: 'whisper', 'wav2vec2', or 'mms'
            model_name: Model identifier
            device: Device string
            lm_path: Optional path to KenLM file
        """
        if model_type not in ["whisper", "wav2vec2", "mms"]:
            raise ValueError(f"model_type must be 'whisper', 'wav2vec2', or 'mms', got: {model_type}")

        self.model_type = model_type
        self.model_name = model_name
        self.lm_path = lm_path  # ✅ Store LM path
        self.device = device if device else (
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.model = None

    def load_model(self):
        """Load the appropriate model."""
        if self.model_type == "whisper":
            print(f"Loading Whisper model '{self.model_name}' on {self.device}...")
            try:
                self.model = whisper.load_model(self.model_name, device=self.device)
                print("Model loaded successfully.")
            except Exception as e:
                raise ValueError(f"Failed to load Whisper model: {e}")

        elif self.model_type == "wav2vec2":
            try:
                # ✅ Pass lm_path
                self.model = Wav2Vec2Model(model_name=self.model_name, device=self.device, lm_path=self.lm_path)
            except Exception as e:
                raise ValueError(f"Failed to load Wav2Vec2 model: {e}") from e

        elif self.model_type == "mms":
            try:
                # ✅ Pass lm_path
                self.model = MMSModel(model_name=self.model_name, device=self.device, lm_path=self.lm_path)
            except Exception as e:
                raise ValueError(f"Failed to load MMS model: {e}") from e

    def _get_transcription(self, audio_path: Path) -> str:
        if self.model_type == "whisper":
            audio = whisper.load_audio(str(audio_path))
            result = self.model.transcribe(audio, language="de")
            return result['text']
        elif self.model_type == "wav2vec2":
            result = self.model.transcribe(audio_path, language="de")
            return result['text']
        elif self.model_type == "mms":
            result = self.model.transcribe(audio_path, language="deu")
            return result['text']

    def evaluate_dataset(self, metadata_path: str, audio_base_path: Path = None, limit: int = None) -> Dict:
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

        print(f"Processing {len(df)} samples...")

        for idx, row in df.iterrows():
            reference = row['sentence']
            accent = row['accent']
            
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

            try:
                if not audio_path.exists():
                    failed_samples += 1
                    continue

                hypothesis = self._get_transcription(audio_path)

                reference_norm = reference.lower().strip()
                hypothesis_norm = hypothesis.lower().strip()

                wer = metrics.calculate_wer(reference_norm, hypothesis_norm)
                cer = metrics.calculate_cer(reference_norm, hypothesis_norm)
                bleu = metrics.calculate_bleu_score(reference_norm, hypothesis_norm)
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

        if not results:
            return {
                'overall_wer': 0.0, 'overall_cer': 0.0, 'overall_bleu': 0.0,
                'per_dialect_wer': {}, 'per_dialect_cer': {}, 'per_dialect_bleu': {},
                'total_samples': 0, 'failed_samples': failed_samples, 'samples': []
            }

        overall_wer = sum(r['wer'] for r in results) / len(results)
        
        references = [r['reference'] for r in results]
        hypotheses = [r['hypothesis'] for r in results]
        cer_result = metrics.batch_cer(references, hypotheses)
        overall_cer = cer_result['overall_cer']
        
        bleu_result = metrics.batch_bleu(references, hypotheses)
        overall_bleu = bleu_result['overall_bleu']

        # Aggregate per-dialect metrics
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
            'samples': results[:5]
        }