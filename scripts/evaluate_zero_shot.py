import sys
import os
from pathlib import Path
from datetime import datetime
from src.config import DATA_DIR, RESULTS_DIR
from src.evaluation.evaluator import ASREvaluator
from src.utils.file_utils import save_results_json, save_results_csv
from src.utils.logging_config import setup_logger

# filepath: /Users/tillermold/Desktop/CODE/Synthesis Semester/Capstone_Project/cp-swiss-german-asr/scripts/evaluate_zero_shot.py

# Add project root to sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def main():
    """
    Zero-shot evaluation script:
    Loads German-adapted ASR model, runs inference on Swiss-German test set,
    computes WER/CER/BLEU, and saves results to CSV/JSON.
    """
    # Model config: German-adapted Wav2Vec2 (no Swiss-German fine-tuning)
    model_type = "wav2vec2"
    model_name = "facebook/wav2vec2-large-xlsr-53-german"

    # Test set path (Swiss-German)
    test_path = DATA_DIR / "metadata" / "test.tsv"

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / "metrics" / f"zero_shot_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    logger = setup_logger("evaluate_zero_shot", str(output_dir / "zero_shot.log"))

    logger.info(f"Zero-shot evaluation: {model_type} - {model_name}")
    logger.info(f"Test set: {test_path}")
    logger.info(f"Output dir: {output_dir}")

    if not test_path.exists():
        logger.error(f"Test file not found: {test_path}")
        print(f"❌ Error: Test file not found at {test_path}")
        return

    # Load model
    evaluator = ASREvaluator(model_type=model_type, model_name=model_name)
    evaluator.load_model()

    # Run evaluation
    logger.info("Running zero-shot evaluation...")
    results = evaluator.evaluate_dataset(metadata_path=str(test_path))

    # Save results
    json_path = output_dir / "zero_shot_results.json"
    csv_path = output_dir / "zero_shot_results.csv"
    save_results_json(results, str(json_path), "zero_shot")
    save_results_csv(results, str(csv_path))

    # Print summary
    print("\nZero-shot Evaluation Results")
    print(f"  Overall WER:  {results['overall_wer']:.4f}")
    print(f"  Overall CER:  {results['overall_cer']:.4f}")
    print(f"  Overall BLEU: {results['overall_bleu']:.4f}")
    print(f"  Samples:      {results['total_samples']}")
    print(f"  Failed:       {results['failed_samples']}")
    print(f"\nResults saved to: {output_dir}")

    logger.info(f"✓ Zero-shot evaluation complete")
    logger.info(f"  Overall WER: {results['overall_wer']:.4f}")
    logger.info(f"  Overall CER: {results['overall_cer']:.4f}")
    logger.info(f"  Overall BLEU: {results['overall_bleu']:.4f}")
    logger.info(f"  Samples processed: {results['total_samples']}")
    logger.info(f"  Failed samples: {results['failed_samples']}")

if __name__ == "__main__":
    main()