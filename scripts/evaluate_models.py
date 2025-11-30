import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path FIRST (before importing from src)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# NOW import from src (after path is fixed)
from src.config import FHNW_SWISS_GERMAN_ROOT, MODELS_DIR, DATA_DIR, RESULTS_DIR
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
    "whisper-large": {"type": "whisper", "name": "large"},
    "whisper-large-v2": {"type": "whisper", "name": "large-v2"},
    "whisper-large-v3": {"type": "whisper", "name": "large-v3"},  # pinned version: identical to "whisper-large"
    "whisper-large-v3-turbo": {"type": "whisper", "name": "large-v3-turbo"},
    
    # Wav2Vec2 models
    ## German model
    "wav2vec2-german": {"type": "wav2vec2", "name": "jonatasgrosman/wav2vec2-large-xlsr-53-german"},
    "wav2vec2-german-1b": {"type": "wav2vec2", "name": "jonatasgrosman/wav2vec2-xls-r-1b-german"},
    # ✅ Added LM support here
    "wav2vec2-german-with-lm": {
        "type": "wav2vec2", 
        "name": "aware-ai/wav2vec2-large-xlsr-53-german-with-lm", 
        "lm_path": str(MODELS_DIR / "lm" / "kenLM.arpa") 
    },

    # ✅ Added adapted German model
    "wav2vec2-ger-nl-adapted": {
        "type": "wav2vec2",
        "name": str(MODELS_DIR / "adapted" / "wav2vec2-german-adapted"),
        "lm_path": str(MODELS_DIR / "adapted" / "wav2vec2-german-adapted" / "language_model" / "KenLM.arpa")
    },
    # Dutch model
    "wav2vec2-dutch-pretrained": {"type": "wav2vec2", "name": "facebook/wav2vec2-large-xlsr-53-dutch"},
    ## Multilingual model
    "wav2vec2-multi-56": {"type": "wav2vec2", "name": "voidful/wav2vec2-xlsr-multilingual-56"},

    # MMS models
    "mms-1b-all": {"type": "mms", "name": "facebook/mms-1b-all"},
    # ⚠️ REMOVED: MMS models have vocab mismatch with KenLM decoders
    # "mms-1b-all-lm": {"type": "mms", "name": "facebook/mms-1b-all", "lm_path": str(MODELS_DIR / "lm" / "kenLM.arpa")},
    "mms-1b-l1107": {"type": "mms", "name": "facebook/mms-1b-l1107"},
    # ⚠️ REMOVED: MMS models have vocab mismatch with KenLM decoders
    # "mms-1b-l1107-lm": {"type": "mms", "name": "facebook/mms-1b-l1107", "lm_path": str(MODELS_DIR / "lm" / "kenLM.arpa")}
    
    
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
        default=str(DATA_DIR / "metadata" / "test.tsv"),
        help="Path to test metadata TSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR / "metrics"),
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of samples to evaluate"
    )
    parser.add_argument(
        "--experiment-type",
        type=str,
        choices=["zero-shot", "fine-tuned", "standard"],
        default="standard",
        help="Type of experiment (e.g., zero-shot, fine-tuned, standard)"
    )
    # ✅ New CLI argument for LM override
    parser.add_argument(
        "--lm-path",
        type=str,
        default=None,
        help="Optional path to KenLM file (overrides registry)"
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger("evaluate_models", "logs/evaluation.log")
    
    # Create timestamp-based subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.experiment_type and args.experiment_type != "standard":
        output_dir = Path(args.output_dir) / f"{args.experiment_type}_{timestamp}"
    else:
        output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting evaluation run: {timestamp}")
    logger.info(f"Experiment type: {args.experiment_type}")
    logger.info(f"Models to evaluate: {args.models}")
    logger.info(f"LM Path override: {args.lm_path}")
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
                
                # ✅ RESOLVE LM PATH: Prioritize CLI arg > Registry arg > None
                registry_lm_path = config.get("lm_path")
                final_lm_path = args.lm_path if args.lm_path else registry_lm_path
                
            else:
                logger.warning(f"Unknown model: {model_spec}")
                print(f"⚠️  Warning: Unknown model: {model_spec}, skipping...")
                continue
            
            # Create evaluator with LM path
            evaluator = ASREvaluator(
                model_type=model_type,
                model_name=model_name,
                lm_path=final_lm_path  # ✅ Pass resolved path
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
    else:
        print("❌ No models were successfully evaluated")
        sys.exit(1)

if __name__ == "__main__":
    main()