import json
import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
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
    python scripts/analyze_errors.py --top_percent 0.1
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

def analyze_single_result(
    file_path: Path, 
    analyzer: ErrorAnalyzer, 
    top_percent: float
) -> Dict[str, Any]:
    """
    Process a single result file to extract error patterns and stats.
    """
    logger.info(f"Processing {file_path.name}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    model_name = data.get('model_name', 'unknown')
    samples = data.get('samples', [])
    
    if not samples:
        logger.warning(f"No samples found in {file_path}")
        return None

    # 1. Dialect-specific Analysis (Confusions, Error Rates)
    # This uses ErrorAnalyzer.analyze_by_dialect which internally uses jiwer
    dialect_stats = analyzer.analyze_by_dialect(samples)
    
    # 2. Global Error Distribution
    # Calculate overall substitution/deletion/insertion rates
    global_alignments = [analyzer.get_alignment(s['reference'], s['hypothesis']) for s in samples]
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
    # Sort descending by WER
    sorted_samples = sorted(samples, key=lambda x: x.get('wer', 0), reverse=True)
    cutoff_index = int(len(samples) * top_percent)
    # Ensure at least a few samples if dataset is small
    cutoff_index = max(cutoff_index, min(5, len(samples)))
    worst_samples = sorted_samples[:cutoff_index]
    
    # Enrich worst samples with readable alignments
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save full hierarchical analysis (JSON)
    json_path = output_dir / f"analysis_{model_name}_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    logger.info(f"  -> Saved detailed analysis: {json_path.name}")

    # 2. Save worst samples (CSV) for easy inspection
    worst_samples = analysis['worst_samples']
    if worst_samples:
        # Flatten the dictionary for CSV
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
        csv_path = output_dir / f"worst_samples_{model_name}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"  -> Saved worst samples CSV: {csv_path.name}")

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
        default=RESULTS_DIR / "analysis",
        help="Directory to save analysis reports"
    )
    parser.add_argument(
        "--top_percent", 
        type=float, 
        default=0.1,
        help="Fraction of worst samples to extract (default: 0.1 for 10%)"
    )
    
    args = parser.parse_args()
    
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
    
    summary_comparison = {}

    for file_path in result_files:
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
        except Exception as e:
            logger.error(f"Failed to analyze {file_path.name}: {e}", exc_info=True)

    # Save comparison summary
    if summary_comparison:
        summary_path = args.output_dir / "model_comparison_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_comparison, f, indent=2)
        logger.info(f"Saved model comparison summary to {summary_path}")

    logger.info("Analysis complete.")

if __name__ == "__main__":
    main()