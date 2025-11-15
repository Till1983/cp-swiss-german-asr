import sys
import os
from pathlib import Path
import argparse
from datetime import datetime
from src.config import FHNW_SWISS_GERMAN_ROOT, MODELS_DIR

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.evaluation.evaluator import ASREvaluator
from src.utils.file_utils import save_results_json, save_results_csv
from src.utils.logging_config import setup_logger

# Model Registry - Maps shorthand names to model configurations
MODEL_REGISTRY = {
    # Whisper models (version-pinned for reproducibility)
    "whisper-tiny": {"type": "whisper", "name": "tiny"},
    "whisper-base": {"type": "whisper", "name": "base"},
    "whisper-small": {"type": "whisper", "name": "small"},
    "whisper-medium": {"type": "whisper", "name": "medium"},
    "whisper-large-v2": {"type": "whisper", "name": "large-v2"},
    "whisper-large-v3": {"type": "whisper", "name": "large-v3"},  # Current best, pinned version
    "whisper-large-v3-turbo": {"type": "whisper", "name": "large-v3-turbo"},
    
    # Wav2Vec2 models
    ## German model
    "wav2vec2-german": {"type": "wav2vec2", "name": "facebook/wav2vec2-large-xlsr-53-german"},
    ## Multilingual model
    "wav2vec2-multi-56": {"type": "wav2vec2", "name": "voidful/wav2vec2-xlsr-multilingual-56"},

    # MMS models
    "mms-1b-all": {"type": "mms", "name": "facebook/mms-1b-all"},  # 1000+ languages
    "mms-1b-l1107": {"type": "mms", "name": "facebook/mms-1b-l1107"},  # 1107 languages
}

def main():
    """Main evaluation script for ASR models on Swiss German test set."""
    parser = argparse.ArgumentParser(
        description="Evaluate ASR models on Swiss German test set"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["whisper-base", "wav2vec2-german"],
        help=f"List of models to evaluate. Available: {', '.join(MODEL_REGISTRY.keys())}"
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="data/metadata/test.tsv",
        help="Path to test metadata TSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/metrics",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of samples to evaluate"
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger("evaluate_models", "logs/evaluation.log")
    
    # Create timestamp-based subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting evaluation run: {timestamp}")
    logger.info(f"Models to evaluate: {args.models}")
    logger.info(f"Test set: {args.test_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Validate test set exists
    test_path = Path(args.test_path)
    if not test_path.exists():
        logger.error(f"Test file not found: {test_path}")
        print(f"❌ Error: Test file not found at {test_path}")
        sys.exit(1)
    
    # Store results for summary table
    all_results = {}
    
    # Evaluate each model
    for model_spec in args.models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating model: {model_spec}")
        logger.info(f"{'='*60}")
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_spec}")
        print(f"{'='*60}")
        
        try:
            # Look up model in registry
            if model_spec in MODEL_REGISTRY:
                config = MODEL_REGISTRY[model_spec]
                model_type = config["type"]
                model_name = config["name"]
            else:
                logger.warning(f"Unknown model: {model_spec}. Available models: {', '.join(MODEL_REGISTRY.keys())}")
                print(f"⚠️  Warning: Unknown model: {model_spec}, skipping...")
                print(f"   Available models: {', '.join(MODEL_REGISTRY.keys())}")
                continue
            
            # Create evaluator
            evaluator = ASREvaluator(
                model_type=model_type,
                model_name=model_name
            )
            
            # Load model
            evaluator.load_model()
            
            # Run evaluation
            logger.info(f"Running evaluation on test set...")
            results = evaluator.evaluate_dataset(
                metadata_path=str(test_path),
                limit=args.limit
            )
            
            # Store results
            all_results[model_spec] = results
            
            # Save results
            json_path = output_dir / f"{model_spec}_results.json"
            csv_path = output_dir / f"{model_spec}_results.csv"
            
            save_results_json(results, str(json_path), model_spec)
            save_results_csv(results, str(csv_path))
            
            logger.info(f"✓ Model {model_spec} evaluation complete")
            logger.info(f"  Overall WER: {results['overall_wer']:.4f}")
            logger.info(f"  Overall CER: {results['overall_cer']:.4f}")
            logger.info(f"  Overall BLEU: {results['overall_bleu']:.4f}")
            logger.info(f"  Samples processed: {results['total_samples']}")
            logger.info(f"  Failed samples: {results['failed_samples']}")
            
            print(f"✓ Model {model_spec} evaluation complete")
            print(f"  Overall WER: {results['overall_wer']:.4f}")
            print(f"  Overall CER: {results['overall_cer']:.4f}")
            print(f"  Overall BLEU: {results['overall_bleu']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error evaluating {model_spec}: {e}", exc_info=True)
            print(f"❌ Error evaluating {model_spec}: {e}")
            print("Continuing with remaining models...")
            continue
    
    # Print summary table
    if all_results:
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"{'Model':<30} {'WER':<10} {'CER':<10} {'BLEU':<10} {'Samples':<10}")
        print(f"{'-'*80}")
        
        for model_spec, results in all_results.items():
            print(f"{model_spec:<30} "
                  f"{results['overall_wer']:<10.4f} "
                  f"{results['overall_cer']:<10.4f} "
                  f"{results['overall_bleu']:<10.4f} "
                  f"{results['total_samples']:<10}")
        
        print(f"{'='*80}")
        print(f"\nResults saved to: {output_dir}")
        
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)
        for model_spec, results in all_results.items():
            logger.info(f"{model_spec}: WER={results['overall_wer']:.4f}, "
                       f"CER={results['overall_cer']:.4f}, "
                       f"BLEU={results['overall_bleu']:.4f}, "
                       f"Samples={results['total_samples']}")
    else:
        print("❌ No models were successfully evaluated")
        logger.error("No models were successfully evaluated")
        sys.exit(1)


if __name__ == "__main__":
    main()