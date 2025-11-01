import whisper
import torch
import pandas as pd
from pathlib import Path
from typing import Dict
from src.evaluation import metrics


class ASREvaluator:
    """
    ASR Evaluator for Whisper models on Swiss German audio datasets.
    """
    
    def __init__(self, model_name: str = "base", device: str = None):
        """
        Initialize the ASR Evaluator.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on ("cuda" or "cpu"). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
    def load_model(self):
        """
        Load the Whisper model onto the specified device.
        """
        print(f"Loading Whisper model '{self.model_name}' on {self.device}...")
        self.model = whisper.load_model(self.model_name, device=self.device)
        print("Model loaded successfully.")
        
    def evaluate_dataset(self, metadata_path: str, audio_base_path: str = "data/raw/fhnw-swiss-german-corpus/clips", limit: int = None) -> Dict:
        """
        Evaluate the model on a dataset defined by a TSV metadata file.
        
        Args:
            metadata_path: Path to TSV file with columns: path, sentence, accent
            audio_base_path: Base directory where audio files are stored
            limit: Optional limit on number of samples to process
            
        Returns:
            Dictionary containing:
                - overall_wer: Overall WER across all samples
                - per_dialect_wer: Dictionary of WER per dialect/accent
                - total_samples: Total number of samples processed
                - failed_samples: Number of samples that failed
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        audio_base_path = Path(audio_base_path)
        
        # Read metadata
        try:
            df = pd.read_csv(metadata_path, sep='\t')
        except Exception as e:
            raise ValueError(f"Failed to read metadata file: {e}")
        
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
            # Construct full audio path by combining base path with filename
            audio_filename = Path(row['path']).name  # Extract just the filename
            audio_path = audio_base_path / audio_filename
            
            reference = row['sentence']
            accent = row['accent']
            
            try:
                # Check if audio file exists
                if not audio_path.exists():
                    print(f"Warning: Audio file not found: {audio_path}")
                    failed_samples += 1
                    continue
                
                # Load audio
                audio = whisper.load_audio(str(audio_path))
                
                # Transcribe
                result = self.model.transcribe(audio, language="de")
                hypothesis = result['text']
                
                # Calculate WER
                wer = metrics.calculate_wer(reference, hypothesis)
                
                results.append({
                    'accent': accent,
                    'wer': wer,
                    'reference': reference,
                    'hypothesis': hypothesis
                })
                
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                failed_samples += 1
                continue
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(df)} samples...")
        
        # Calculate overall WER
        if not results:
            return {
                'overall_wer': 0.0,
                'per_dialect_wer': {},
                'total_samples': 0,
                'failed_samples': failed_samples
            }
        
        overall_wer = sum(r['wer'] for r in results) / len(results)
        
        # Calculate per-dialect WER
        per_dialect_wer = {}
        results_df = pd.DataFrame(results)
        for accent in results_df['accent'].unique():
            accent_results = results_df[results_df['accent'] == accent]
            per_dialect_wer[accent] = accent_results['wer'].mean()
        
        return {
            'overall_wer': overall_wer,
            'per_dialect_wer': per_dialect_wer,
            'total_samples': len(results),
            'failed_samples': failed_samples
        }