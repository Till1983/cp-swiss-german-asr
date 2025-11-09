import json
import pandas as pd
from pathlib import Path
from typing import Dict
from datetime import datetime

def save_results_json(results: Dict, output_path: str, model_name: str) -> None:
    """
    Save evaluation results to JSON file with timestamp and model metadata.
    
    Args:
        results: Dictionary containing evaluation results
        output_path: Path where JSON file should be saved
        model_name: Name of the model being evaluated
        
    Raises:
        IOError: If file cannot be written
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_data = {
            'timestamp': timestamp,
            'model_name': model_name,
            'results': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        print(f"✓ JSON results saved to {output_path}")
        
    except Exception as e:
        raise IOError(f"Failed to save results to JSON: {e}") from e


def save_results_csv(results: Dict, output_path: str) -> None:
    """
    Convert per-dialect metrics (WER, CER, BLEU) to CSV format.
    
    Args:
        results: Dictionary containing evaluation results with per-dialect metrics
        output_path: Path where CSV file should be saved
        
    Raises:
        ValueError: If results don't contain required metrics
        IOError: If file cannot be written
    """
    try:
        # Validate all required keys
        required_per_dialect_keys = {'per_dialect_wer', 'per_dialect_cer', 'per_dialect_bleu'}
        required_overall_keys = {'overall_wer', 'overall_cer', 'overall_bleu'}
        
        missing_per_dialect = required_per_dialect_keys - set(results.keys())
        missing_overall = required_overall_keys - set(results.keys())
        
        if missing_per_dialect or missing_overall:
            missing = missing_per_dialect | missing_overall
            raise ValueError(
                f"Results dictionary must contain keys: {required_per_dialect_keys | required_overall_keys}. "
                f"Missing: {missing}"
            )
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get all unique dialects (should be consistent across metrics)
        dialects = set(results['per_dialect_wer'].keys())
        
        # Build comprehensive per-dialect data
        data = []
        for dialect in sorted(dialects):
            data.append({
                'dialect': dialect,
                'wer': results['per_dialect_wer'].get(dialect, None),
                'cer': results['per_dialect_cer'].get(dialect, None),
                'bleu': results['per_dialect_bleu'].get(dialect, None)
            })
        
        # Add overall metrics as a summary row
        data.append({
            'dialect': 'OVERALL',
            'wer': results.get('overall_wer', None),
            'cer': results.get('overall_cer', None),
            'bleu': results.get('overall_bleu', None)
        })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        print(f"✓ CSV results saved to {output_path}")
        
    except Exception as e:
        raise IOError(f"Failed to save results to CSV: {e}") from e


def ensure_log_directory(log_path: str) -> None:
    """
    Ensure the log directory exists.
    
    Args:
        log_path: Path to the log file
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)