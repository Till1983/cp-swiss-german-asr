import pandas as pd
from typing import Literal


METRIC_THRESHOLDS = {
    'wer': {
        'excellent': (0, 30),
        'good': (30, 50),
        'poor': (50, float('inf'))
    },
    'cer': {
        'excellent': (0, 15),
        'good': (15, 35),
        'poor': (35, float('inf'))
    },
    'bleu': {
        'excellent': (50, 100),
        'good': (30, 50),
        'poor': (0, 30)
    }
}


def _get_performance_category(value: float, metric: str) -> str:
    """
    Determine performance category based on metric value.
    
    Args:
        value: Metric value
        metric: Metric name ('wer', 'cer', or 'bleu')
        
    Returns:
        Performance category: 'excellent', 'good', or 'poor'
    """
    thresholds = METRIC_THRESHOLDS[metric]
    
    for category, (low, high) in thresholds.items():
        if low <= value < high:
            return category
    
    return 'poor'


def _apply_color_formatting(val: float, metric: str) -> str:
    """
    Apply color formatting to cell based on performance.
    
    Args:
        val: Cell value
        metric: Metric name
        
    Returns:
        CSS style string
    """
    category = _get_performance_category(val, metric)
    
    colors = {
        'excellent': 'background-color: #90EE90',  # Light green
        'good': 'background-color: #FFFFE0',        # Light yellow
        'poor': 'background-color: #FFB6C6'         # Light red
    }
    
    return colors.get(category, '')


def compare_models(
    df: pd.DataFrame,
    selected_metric: Literal['wer', 'cer', 'bleu'] = 'wer',
    sort_ascending: bool = True
) -> pd.DataFrame:
    """
    Compare models across dialects with color-coded performance.
    
    Args:
        df: DataFrame with columns ['model', 'dialect', 'wer', 'cer', 'bleu']
        selected_metric: Metric to use for sorting
        sort_ascending: Sort direction (True for ascending, False for descending)
        
    Returns:
        Styled DataFrame pivot table
    """
    # Filter out OVERALL rows for comparison
    df_filtered = df[df['dialect'] != 'OVERALL'].copy()
    
    # Create pivot table
    pivot = df_filtered.pivot_table(
        index='model',
        columns='dialect',
        values=selected_metric,
        aggfunc='mean'
    )
    
    # Add overall average column
    pivot['OVERALL'] = pivot.mean(axis=1)
    
    # Sort by selected metric
    pivot = pivot.sort_values('OVERALL', ascending=sort_ascending)
    
    # Apply color formatting
    styled = pivot.style.map(
        lambda x: _apply_color_formatting(x, selected_metric)
    ).format("{:.2f}")
    
    return styled