"""
Script to inspect and compare model transcription results.
"""
import json
import sys
from pathlib import Path
from typing import List, Dict


def load_results(results_path: Path) -> Dict:
    """Load results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def print_transcriptions(results: Dict, model_name: str, num_samples: int = 10):
    """Print transcriptions for inspection."""
    print(f"\n{'=' * 80}")
    print(f"MODEL: {model_name}")
    print(f"Overall WER: {results['results']['overall_wer']:.2f}%")
    print(f"Overall CER: {results['results']['overall_cer']:.2f}%")
    print(f"{'=' * 80}\n")
    
    samples = results.get('samples', [])[:num_samples]
    
    for i, sample in enumerate(samples, 1):
        print(f"Sample {i} (Dialect: {sample.get('dialect', 'Unknown')})")
        print(f"  File: {sample.get('audio_file', 'Unknown')}")
        print(f"  Reference: {sample['reference']}")
        print(f"  Predicted: {sample['hypothesis']}")
        print(f"  WER: {sample['wer']:.2f}% | CER: {sample['cer']:.2f}%")
        
        # Check if transcription is empty or in wrong language
        pred = sample['hypothesis'].lower()
        if not pred.strip():
            print("  ⚠️  WARNING: Empty transcription!")
        elif detect_language_hints(pred):
            print(f"  ⚠️  WARNING: Possible language mismatch detected")
        
        print()


def detect_language_hints(text: str) -> str:
    """Detect potential language issues in transcription."""
    text_lower = text.lower()
    
    # Common English words that shouldn't appear in Swiss German
    english_indicators = ['the', 'and', 'is', 'are', 'was', 'were', 'have', 'has']
    english_count = sum(1 for word in english_indicators if f' {word} ' in f' {text_lower} ')
    
    if english_count >= 2:
        return "English"
    
    # Check for French
    french_indicators = ['le', 'la', 'les', 'et', 'est', 'sont']
    french_count = sum(1 for word in french_indicators if f' {word} ' in f' {text_lower} ')
    
    if french_count >= 2:
        return "French"
    
    return ""


def compare_models(results_dir: Path, num_samples: int = 5):
    """Compare transcriptions across multiple models."""
    print(f"\n{'=' * 80}")
    print("COMPARING MODELS SIDE-BY-SIDE")
    print(f"{'=' * 80}\n")
    
    # Load all result files in directory
    result_files = list(results_dir.glob("*_results.json"))
    
    if not result_files:
        print(f"No result files found in {results_dir}")
        return
    
    # Load all results
    all_results = {}
    for file in result_files:
        model_name = file.stem.replace('_results', '')
        all_results[model_name] = load_results(file)
    
    # Get samples from first model as reference
    first_model = list(all_results.keys())[0]
    num_samples = min(num_samples, len(all_results[first_model].get('samples', [])))
    
    for i in range(num_samples):
        print(f"\n{'─' * 80}")
        print(f"SAMPLE {i + 1}")
        print(f"{'─' * 80}")
        
        # Print reference once
        first_sample = all_results[first_model]['samples'][i]
        print(f"\nReference: {first_sample['reference']}")
        print(f"Dialect: {first_sample.get('dialect', 'Unknown')}")
        print(f"File: {first_sample.get('audio_file', 'Unknown')}\n")
        
        # Print each model's transcription
        for model_name, results in all_results.items():
            sample = results['samples'][i]
            print(f"{model_name}:")
            print(f"  → {sample['hypothesis']}")
            print(f"  WER: {sample['wer']:.1f}% | CER: {sample['cer']:.1f}%")


def main():
    """Main inspection function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect ASR model transcription results")
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Path to results directory (e.g., results/metrics/20251113_164317)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Specific model to inspect (optional)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all models side-by-side'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples to show (default: 10)'
    )
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)
    
    if args.compare:
        compare_models(results_dir, args.num_samples)
    elif args.model:
        # Inspect specific model
        results_file = results_dir / f"{args.model}_results.json"
        if not results_file.exists():
            print(f"Error: Results file not found: {results_file}")
            sys.exit(1)
        
        results = load_results(results_file)
        print_transcriptions(results, args.model, args.num_samples)
    else:
        # Inspect all models
        result_files = list(results_dir.glob("*_results.json"))
        for file in result_files:
            model_name = file.stem.replace('_results', '')
            results = load_results(file)
            print_transcriptions(results, model_name, args.num_samples)


if __name__ == "__main__":
    main()