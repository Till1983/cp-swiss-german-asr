import json
import argparse
import logging
import pandas as pd
import sys
import time
import platform
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Try to import jiwer to check version
try:
    import jiwer
    try:
        from importlib.metadata import version
        JIWER_VERSION = version('jiwer')
    except Exception:
        JIWER_VERSION = "unknown"
except ImportError:
    JIWER_VERSION = "unknown"

from src.config import RESULTS_DIR
from src.evaluation.error_analyzer import ErrorAnalyzer


"""
Script to analyze ASR error patterns from evaluation results.

Functionality:
1. Loads evaluation results (JSON) from results/metrics/
2. Uses ErrorAnalyzer to compute detailed error statistics
3. Identifies high-error samples (worst N% by WER)
4. Generates dialect-specific confusion matrices
5. Analyzes WER-BLEU correlation for semantic preservation (NEW)
6. Outputs structured analysis for dashboard integration

Usage:
    python scripts/analyze_errors.py --input_dir results/metrics --top_percent 0.1
"""

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_result_files(metrics_dir: Path) -> List[Path]:
    """Find all result JSON files in the metrics directory."""
    if not metrics_dir.exists():
        logger.warning(f"Metrics directory not found: {metrics_dir}")
        return []
    # Look for files ending in _results.json recursively
    return list(metrics_dir.rglob("*_results.json"))


def log_memory_usage():
    """Helper to log current memory usage if psutil is available."""
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        mem_info = process.memory_info()
        logger.info(f"Memory Usage: {mem_info.rss / 1024 / 1024:.2f} MB")


def analyze_single_result(
    file_path: Path, 
    analyzer: ErrorAnalyzer, 
    top_percent: float
) -> Dict[str, Any]:
    """
    Process a single result file to extract error patterns and stats.
    """
    logger.info(f"Processing {file_path.name}...")
    sys.stdout.flush()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    model_name = data.get('model_name', 'unknown')
    
    # Handle both nested (results.samples) and flat (samples) structures
    results_obj = data.get('results', {})
    samples = results_obj.get('samples', data.get('samples', []))
    
    total_samples = len(samples)
    
    if not samples:
        logger.warning(f"No samples found in {file_path}")
        return None

    logger.info(f"  Total samples: {total_samples}")
    
    # 1. Dialect-specific Analysis
    logger.info("  Computing dialect-specific confusion patterns...")
    sys.stdout.flush()
    dialect_stats = analyzer.analyze_by_dialect(samples)
    
    # 2. Global Alignment Computation
    logger.info("  Computing global alignments (this may take a while)...")
    sys.stdout.flush()
    
    # Progress tracking with dots
    global_alignments = []
    for i, s in enumerate(samples, 1):
        align = analyzer.get_alignment(s['reference'], s['hypothesis'])
        global_alignments.append(align)
        
        # Progress dots every 50 samples
        if i % 50 == 0:
            print('.', end='', flush=True)
        
        # Detailed log every 100 samples
        if i % 100 == 0:
            logger.info(f"  Processed {i}/{total_samples} alignments...")
            sys.stdout.flush()
        
        # Memory check every 200 samples
        if i % 200 == 0 and PSUTIL_AVAILABLE:
            log_memory_usage()
    
    print()  # Newline after dots
    logger.info(f"  ✓ Completed {total_samples} alignments")
    sys.stdout.flush()
    
    # 3. Global Error Distribution
    logger.info("  Computing global error type distribution...")
    sys.stdout.flush()
    
    global_counts = {'substitution': 0, 'deletion': 0, 'insertion': 0, 'correct': 0}
    
    for align in global_alignments:
        counts = analyzer.categorize_errors(align)
        for k in global_counts:
            global_counts[k] += counts.get(k, 0)
    
    total_ops = sum(global_counts.values())
    global_dist = {
        k: (v / total_ops * 100) if total_ops > 0 else 0.0 
        for k, v in global_counts.items()
    }
    
    # 4. WER-BLEU Correlation Analysis (NEW)
    logger.info("  Analyzing WER-BLEU correlation for semantic preservation...")
    sys.stdout.flush()
    
    wer_bleu_analysis = analyzer.analyze_wer_bleu_correlation(
        samples,
        wer_threshold=50.0,
        bleu_threshold=40.0
    )
    
    logger.info(f"  ✓ Found {wer_bleu_analysis['summary']['high_wer_high_bleu_count']} samples with semantic preservation (high WER + high BLEU)")
    sys.stdout.flush()
    
    # 5. Extract Worst Samples (Top N% by WER)
    logger.info(f"  Identifying worst {top_percent*100:.0f}% samples by WER...")
    sys.stdout.flush()
    
    sorted_samples = sorted(samples, key=lambda x: x.get('wer', 0), reverse=True)
    cutoff_index = int(len(samples) * top_percent)
    cutoff_index = max(cutoff_index, min(5, len(samples)))
    worst_samples = sorted_samples[:cutoff_index]
    
    logger.info(f"  Enriching {len(worst_samples)} worst samples with alignments...")
    sys.stdout.flush()
    
    enriched_worst = []
    for s in worst_samples:
        align = analyzer.get_alignment(s['reference'], s['hypothesis'])
        s_copy = s.copy()
        s_copy['alignment_readable'] = analyzer.format_alignment_readable(align)
        s_copy['error_counts'] = analyzer.categorize_errors(align)
        enriched_worst.append(s_copy)

    return {
        'meta': {
            'model_name': model_name,
            'source_file': str(file_path.name),
            'timestamp': data.get('timestamp'),
            'total_samples': len(samples)
        },
        'global_metrics': analyzer.calculate_aggregate_stats(samples),
        'error_distribution_percent': global_dist,
        'wer_bleu_correlation': wer_bleu_analysis,  # NEW
        'dialect_analysis': dialect_stats,
        'worst_samples': enriched_worst
    }


def save_analysis(analysis: Dict[str, Any], output_dir: Path):
    """Save analysis results to JSON and CSV formats."""
    model_name = analysis['meta']['model_name']
    
    # Use simple filenames without redundant timestamps
    # (output_dir is already timestamped by shell script)
    json_path = output_dir / f"analysis_{model_name}.json"
    csv_path = output_dir / f"worst_samples_{model_name}.csv"
    
    # 1. Save full hierarchical analysis (JSON)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    logger.info(f"  -> Saved detailed analysis: {json_path.name}")
    sys.stdout.flush()

    # 2. Save worst samples (CSV) for easy inspection
    worst_samples = analysis['worst_samples']
    if worst_samples:
        flat_samples = []
        for s in worst_samples:
            flat_s = {
                'dialect': s.get('dialect'),
                'wer': s.get('wer'),
                'cer': s.get('cer'),
                'bleu': s.get('bleu', 0.0),  # NEW
                'reference': s.get('reference'),
                'hypothesis': s.get('hypothesis'),
                'substitutions': s['error_counts']['substitution'],
                'deletions': s['error_counts']['deletion'],
                'insertions': s['error_counts']['insertion'],
                'alignment_viz': s['alignment_readable']
            }
            flat_samples.append(flat_s)
            
        df = pd.DataFrame(flat_samples)
        df.to_csv(csv_path, index=False)
        logger.info(f"  -> Saved worst samples CSV: {csv_path.name}")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Analyze ASR error patterns and generate reports.")
    parser.add_argument(
        "--input_dir", 
        type=Path, 
        default=RESULTS_DIR / "metrics",
        help="Directory containing *_results.json files"
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        default=RESULTS_DIR / "error_analysis",
        help="Directory to save analysis reports"
    )
    parser.add_argument(
        "--top_percent", 
        type=float, 
        default=0.1,
        help="Fraction of worst samples to extract (default: 0.1 for 10%%)"
    )
    
    args = parser.parse_args()
    
    # Environment Logging
    logger.info("=" * 60)
    logger.info("Starting Error Analysis Script")
    logger.info("=" * 60)
    logger.info(f"Python Version: {platform.python_version()}")
    logger.info(f"Jiwer Version: {JIWER_VERSION}")
    logger.info(f"Psutil Available: {PSUTIL_AVAILABLE}")
    logger.info(f"Input Directory: {args.input_dir}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Top Percent: {args.top_percent * 100:.0f}%")
    sys.stdout.flush()
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = ErrorAnalyzer()
    
    # Find files
    result_files = load_result_files(args.input_dir)
    if not result_files:
        logger.error(f"No result files found in {args.input_dir}")
        return

    logger.info(f"Found {len(result_files)} result files to analyze.")
    sys.stdout.flush()
    
    # Process each file
    all_analyses = []
    
    for file_path in result_files:
        try:
            result = analyze_single_result(file_path, analyzer, args.top_percent)
            if result:
                all_analyses.append(result)
                save_analysis(result, args.output_dir)
                
                # Log summary statistics
                metrics = result['global_metrics']
                corr = result['wer_bleu_correlation']['summary']
                logger.info(f"  Summary for {result['meta']['model_name']}:")
                logger.info(f"    Overall WER: {metrics['mean_wer']:.2f}%")
                logger.info(f"    Overall CER: {metrics['mean_cer']:.2f}%")
                logger.info(f"    Overall BLEU: {metrics['mean_bleu']:.2f}")
                logger.info(f"    Semantic preservation rate: {corr['semantic_preservation_rate']:.2f}%")
                sys.stdout.flush()
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            continue
    
    # Generate comparison summary
    if all_analyses:
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        for analysis in all_analyses:
            model_name = analysis['meta']['model_name']
            comparison['models'][model_name] = {
                'mean_wer': analysis['global_metrics']['mean_wer'],
                'mean_cer': analysis['global_metrics']['mean_cer'],
                'mean_bleu': analysis['global_metrics']['mean_bleu'],
                'total_samples': analysis['meta']['total_samples'],
                'semantic_preservation_rate': analysis['wer_bleu_correlation']['summary']['semantic_preservation_rate']
            }
        
        comparison_path = args.output_dir / "model_comparison_summary.json"
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✓ Saved model comparison: {comparison_path}")
        sys.stdout.flush()
    
    logger.info("\n" + "=" * 60)
    logger.info("Error Analysis Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()