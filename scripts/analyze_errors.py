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
5. Outputs structured analysis for dashboard integration

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
    
    # 2. Global Error Distribution
    logger.info(f"  Computing alignments for {total_samples} samples...")
    sys.stdout.flush()
    
    global_alignments = []
    
    # Progress loop with heartbeat
    for i, s in enumerate(samples):
        global_alignments.append(analyzer.get_alignment(s['reference'], s['hypothesis']))
        
        # Heartbeat: print dot every 50 samples
        if (i + 1) % 50 == 0:
            print(".", end="", flush=True)
            
        # Progress log every 100 samples
        if (i + 1) % 100 == 0:
            print()  # Clear dots line
            progress_pct = ((i + 1) / total_samples) * 100
            logger.info(f"    Aligned {i + 1}/{total_samples} samples ({progress_pct:.1f}%)")
            sys.stdout.flush()
        
        # Memory check every 200 samples
        if (i + 1) % 200 == 0:
            log_memory_usage()

    if total_samples % 100 != 0:  # Ensure final newline after dots
        print()
    
    logger.info("  Categorizing global errors...")
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

    # 3. Extract Worst Samples (Top N% by WER)
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
    
    summary_comparison = {}
    total_start_time = time.time()
    processed_count = 0

    for idx, file_path in enumerate(result_files, 1):
        logger.info("")
        logger.info(f"[{idx}/{len(result_files)}] Starting analysis: {file_path.name}")
        sys.stdout.flush()
        
        model_start_time = time.time()
        
        try:
            analysis = analyze_single_result(file_path, analyzer, args.top_percent)
            if analysis:
                save_analysis(analysis, args.output_dir)
                
                # Collect high-level stats for comparison summary
                model_name = analysis['meta']['model_name']
                summary_comparison[model_name] = {
                    'wer_mean': analysis['global_metrics'].get('mean_wer'),
                    'cer_mean': analysis['global_metrics'].get('mean_cer'),
                    'sub_rate': analysis['error_distribution_percent'].get('substitution'),
                    'del_rate': analysis['error_distribution_percent'].get('deletion'),
                    'ins_rate': analysis['error_distribution_percent'].get('insertion'),
                }
                processed_count += 1
                
                # Log duration and memory
                duration = time.time() - model_start_time
                logger.info(f"Finished {model_name} in {duration:.2f}s")
                log_memory_usage()
                sys.stdout.flush()

        except Exception as e:
            logger.error(f"Failed to analyze {file_path.name}: {e}", exc_info=True)
            sys.stdout.flush()

    # Save comparison summary
    if summary_comparison:
        summary_path = args.output_dir / "model_comparison_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_comparison, f, indent=2)
        logger.info(f"Saved model comparison summary to {summary_path.name}")
        sys.stdout.flush()

    total_duration = time.time() - total_start_time
    avg_time = total_duration / processed_count if processed_count > 0 else 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("Analysis Complete")
    logger.info("=" * 60)
    logger.info(f"Total Models Processed: {processed_count}/{len(result_files)}")
    logger.info(f"Total Time: {total_duration:.2f}s ({total_duration/60:.1f}min)")
    logger.info(f"Average Time per Model: {avg_time:.2f}s")
    sys.stdout.flush()


if __name__ == "__main__":
    main()