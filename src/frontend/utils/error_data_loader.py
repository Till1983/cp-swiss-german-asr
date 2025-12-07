"""
Error Analysis Data Loader for Swiss German ASR Dashboard

This module is responsible ONLY for loading and parsing error analysis JSON files.
It does NOT contain visualization logic - that belongs in components.

Responsibilities:
- Load error analysis JSON files with caching
- Parse nested JSON structures into flat, usable formats
- Extract specific data (dialect stats, confusion pairs, worst samples)
- Handle file I/O errors gracefully
"""

import json
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd


@st.cache_data
def load_error_analysis_json(json_path: str) -> Dict[str, Any]:
    """
    Load a single error analysis JSON file with caching.
    
    Args:
        json_path: Path to the analysis JSON file
        
    Returns:
        Dictionary containing the error analysis data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is malformed
    """
    try:
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Error analysis file not found: {json_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON file {json_path}: {e}")
    except Exception as e:
        raise IOError(f"Failed to load error analysis file: {e}") from e


def get_available_error_analyses(
    error_analysis_dir: str = "results/error_analysis"
) -> List[Dict[str, str]]:
    """
    Get list of all available error analysis JSON files.
    
    Searches recursively through timestamped subdirectories.
    
    Args:
        error_analysis_dir: Path to error_analysis directory
        
    Returns:
        List of dictionaries with 'model_name', 'json_path', and 'timestamp' keys
    """
    results = []
    analysis_path = Path(error_analysis_dir)
    
    if not analysis_path.exists():
        return results
    
    try:
        # Find all analysis_*.json files RECURSIVELY (to handle timestamped subdirs)
        for json_file in sorted(analysis_path.rglob("analysis_*.json"), reverse=True):
            # Extract model name from filename
            filename = json_file.stem
            if filename.startswith("analysis_"):
                model_name = filename.replace("analysis_", "")
            else:
                model_name = filename
            
            # Get file timestamp
            timestamp = json_file.stat().st_mtime
            
            results.append({
                'model_name': model_name,
                'json_path': str(json_file),
                'timestamp': timestamp
            })
        
        return results
        
    except FileNotFoundError as e:
        st.warning(f"Error analysis directory not found: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected error scanning error analysis directory: {e}")
        raise


def load_all_error_analyses(
    error_analysis_dir: str = "results/error_analysis",
    model_name: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load all available error analysis JSON files or a specific model's analysis.
    
    Args:
        error_analysis_dir: Path to error_analysis directory
        model_name: Optional specific model to load (None loads all)
        
    Returns:
        Dictionary mapping model names to their analysis data
    """
    analyses = {}
    available = get_available_error_analyses(error_analysis_dir)
    
    for file_info in available:
        file_model_name = file_info['model_name']
        
        # Skip if specific model requested and this isn't it
        if model_name and file_model_name != model_name:
            continue
        
        json_path = file_info['json_path']
        
        try:
            analyses[file_model_name] = load_error_analysis_json(json_path)
        except (FileNotFoundError, ValueError, IOError) as e:
            st.warning(f"Failed to load analysis for {file_model_name}: {e}")
            continue
    
    return analyses


def extract_dialect_statistics(
    analysis_data: Dict[str, Any],
    dialect: str
) -> Optional[Dict[str, Any]]:
    """
    Extract statistics for a specific dialect from error analysis data.
    
    Args:
        analysis_data: Loaded error analysis JSON data
        dialect: Dialect code (e.g., 'BE', 'ZH')
        
    Returns:
        Dictionary with dialect statistics or None if not found
    """
    if 'dialect_analysis' not in analysis_data:
        return None
    
    return analysis_data['dialect_analysis'].get(dialect)


def extract_all_dialect_metrics(analysis_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract per-dialect metrics from error analysis data as DataFrame.
    
    Args:
        analysis_data: Loaded error analysis JSON data
        
    Returns:
        DataFrame with columns: dialect, sample_count, mean_wer, std_wer, 
        mean_cer, mean_bleu, sub_rate, del_rate, ins_rate
    """
    dialect_analysis = analysis_data.get('dialect_analysis', {})
    
    if not dialect_analysis:
        return pd.DataFrame()
    
    rows = []
    for dialect, metrics in dialect_analysis.items():
        error_dist = metrics.get('error_distribution', {})
        
        row = {
            'dialect': dialect,
            'sample_count': metrics.get('sample_count', 0),
            'mean_wer': metrics.get('mean_wer', 0.0),
            'std_wer': metrics.get('std_wer', 0.0),
            'mean_cer': metrics.get('mean_cer', 0.0),
            'mean_bleu': metrics.get('mean_bleu', 0.0),
            'std_bleu': metrics.get('std_bleu', 0.0),
            'substitution_count': error_dist.get('substitution', 0),
            'deletion_count': error_dist.get('deletion', 0),
            'insertion_count': error_dist.get('insertion', 0),
            'correct_count': error_dist.get('correct', 0),
            'sub_rate': error_dist.get('sub_rate', 0.0),
            'del_rate': error_dist.get('del_rate', 0.0),
            'ins_rate': error_dist.get('ins_rate', 0.0),
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def extract_confusion_pairs_raw(
    dialect_stats: Dict[str, Any],
    top_n: int = 20
) -> List[tuple]:
    """
    Extract raw confusion pairs from dialect statistics.
    
    Args:
        dialect_stats: Statistics for specific dialect
        top_n: Maximum number of confusion pairs to return
        
    Returns:
        List of tuples: [(ref_word, hyp_word, count), ...]
    """
    if not dialect_stats or 'top_confusions' not in dialect_stats:
        return []
    
    confusion_pairs = dialect_stats['top_confusions']
    
    if not confusion_pairs:
        return []
    
    result = []
    for pair_data in confusion_pairs[:top_n]:
        if isinstance(pair_data, (list, tuple)) and len(pair_data) >= 2:
            pair, count = pair_data[0], pair_data[1]
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                result.append((pair[0], pair[1], count))
    
    return result


def extract_global_error_distribution(analysis_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract global error distribution percentages.
    
    Args:
        analysis_data: Loaded error analysis JSON data
        
    Returns:
        Dictionary with error types and their percentages
    """
    return analysis_data.get('error_distribution_percent', {})


def extract_aggregate_stats(analysis_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract aggregate statistics from error analysis data.
    
    Args:
        analysis_data: Loaded error analysis JSON data
        
    Returns:
        Dictionary with aggregate statistics (mean_wer, mean_cer, etc.)
    """
    return analysis_data.get('aggregate_stats', {})


def get_worst_samples_path(
    model_name: str,
    error_analysis_dir: str = "results/error_analysis"
) -> Optional[str]:
    """
    Get path to worst_samples CSV for a given model.
    
    Searches recursively through timestamped subdirectories.
    
    Args:
        model_name: Name of the model
        error_analysis_dir: Path to error_analysis directory
        
    Returns:
        Path to CSV file if it exists, None otherwise
    """
    analysis_path = Path(error_analysis_dir)
    
    # Search recursively for worst_samples CSV
    csv_pattern = f"worst_samples_{model_name}.csv"
    matching_files = list(analysis_path.rglob(csv_pattern))
    
    if matching_files:
        # Return the most recent file if multiple exist
        return str(sorted(matching_files, key=lambda p: p.stat().st_mtime, reverse=True)[0])
    
    return None


@st.cache_data
def load_worst_samples(csv_path: str) -> pd.DataFrame:
    """
    Load worst samples CSV file with caching.
    
    Args:
        csv_path: Path to the worst_samples CSV file
        
    Returns:
        DataFrame with worst performing samples
    """
    try:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Worst samples file not found: {csv_path}")
        
        df = pd.read_csv(path)
        return df
        
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Failed to load worst samples: {e}")
        return pd.DataFrame()


# ============================================================================
# AGGREGATE FUNCTIONS FOR CROSS-MODEL COMPARISON
# ============================================================================

def aggregate_model_comparison(
    analyses: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Aggregate error statistics across multiple models for comparison.
    
    Args:
        analyses: Dictionary mapping model names to their analysis data
        
    Returns:
        DataFrame with columns: model, mean_wer, mean_cer, mean_bleu,
        substitution_pct, deletion_pct, insertion_pct, total_samples
    """
    if not analyses:
        return pd.DataFrame()
    
    rows = []
    
    for model_name, data in analyses.items():
        stats = data.get('aggregate_stats', {})
        error_dist = data.get('error_distribution_percent', {})
        
        # Count total samples from dialect analysis
        dialect_analysis = data.get('dialect_analysis', {})
        total_samples = sum(
            d.get('sample_count', 0) 
            for d in dialect_analysis.values()
        )
        
        row = {
            'model': model_name,
            'mean_wer': stats.get('mean_wer', 0.0),
            'median_wer': stats.get('median_wer', 0.0),
            'std_wer': stats.get('std_wer', 0.0),
            'mean_cer': stats.get('mean_cer', 0.0),
            'mean_bleu': stats.get('mean_bleu', 0.0),
            'substitution_pct': error_dist.get('substitution', 0.0),
            'deletion_pct': error_dist.get('deletion', 0.0),
            'insertion_pct': error_dist.get('insertion', 0.0),
            'correct_pct': error_dist.get('correct', 0.0),
            'total_samples': total_samples
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by mean_wer ascending (best first)
    if not df.empty:
        df = df.sort_values('mean_wer', ascending=True).reset_index(drop=True)
    
    return df


def aggregate_dialect_comparison(
    analyses: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Aggregate dialect-level metrics across multiple models.
    
    Args:
        analyses: Dictionary mapping model names to their analysis data
        
    Returns:
        DataFrame with columns: model, dialect, mean_wer, mean_cer, 
        mean_bleu, sample_count, sub_rate, del_rate, ins_rate
    """
    if not analyses:
        return pd.DataFrame()
    
    rows = []
    
    for model_name, data in analyses.items():
        dialect_df = extract_all_dialect_metrics(data)
        
        if not dialect_df.empty:
            dialect_df['model'] = model_name
            rows.append(dialect_df)
    
    if not rows:
        return pd.DataFrame()
    
    combined = pd.concat(rows, ignore_index=True)
    
    # Reorder columns
    cols = ['model', 'dialect'] + [c for c in combined.columns if c not in ['model', 'dialect']]
    combined = combined[cols]
    
    return combined