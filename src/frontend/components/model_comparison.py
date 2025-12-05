import pandas as pd
from typing import Literal, Dict, List, Optional
import plotly.graph_objects as go

from .plotly_charts import (
    create_metric_comparison_chart,
    MODEL_COLORS,
)


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


def _df_to_chart_data(
    df: pd.DataFrame,
    metric: str
) -> Dict[str, Dict[str, float]]:
    """
    Convert DataFrame to chart data format.
    
    Args:
        df: DataFrame with columns ['model', 'dialect', metric]
        metric: Metric column name
        
    Returns:
        Dictionary mapping model names to dialect -> metric value dicts
    """
    data = {}
    df_filtered = df[df['dialect'] != 'OVERALL'].copy()
    
    for model in df_filtered['model'].unique():
        model_df = df_filtered[df_filtered['model'] == model]
        data[model] = dict(zip(model_df['dialect'], model_df[metric]))
    
    return data


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


def create_model_comparison_chart(
    df: pd.DataFrame,
    selected_metric: Literal['wer', 'cer', 'bleu'] = 'wer',
    dialects: Optional[List[str]] = None,
    height: int = 500,
    show_legend: bool = True,
) -> go.Figure:
    """
    Create interactive Plotly chart comparing models across dialects.
    
    Args:
        df: DataFrame with columns ['model', 'dialect', 'wer', 'cer', 'bleu']
        selected_metric: Metric to visualize
        dialects: Optional list of dialects to include
        height: Chart height in pixels
        show_legend: Whether to show legend
        
    Returns:
        Plotly Figure with grouped bar chart
    """
    metric_labels = {
        'wer': 'Word Error Rate',
        'cer': 'Character Error Rate',
        'bleu': 'BLEU Score'
    }
    
    chart_data = _df_to_chart_data(df, selected_metric)
    title = f"{metric_labels.get(selected_metric, selected_metric)} by Dialect and Model"
    
    return create_metric_comparison_chart(
        data=chart_data,
        dialects=dialects,
        metric_name=selected_metric.upper(),
        title=title,
        height=height,
        show_legend=show_legend,
    )


def create_multi_metric_comparison(
    df: pd.DataFrame,
    metrics: List[Literal['wer', 'cer', 'bleu']] = None,
    height: int = 400,
) -> Dict[str, go.Figure]:
    """
    Create multiple comparison charts for different metrics.
    
    Args:
        df: DataFrame with columns ['model', 'dialect', 'wer', 'cer', 'bleu']
        metrics: List of metrics to visualize. Defaults to ['wer', 'cer', 'bleu']
        height: Chart height in pixels
        
    Returns:
        Dictionary mapping metric names to Plotly Figures
    """
    if metrics is None:
        metrics = ['wer', 'cer', 'bleu']
    
    figures = {}
    for metric in metrics:
        figures[metric] = create_model_comparison_chart(
            df=df,
            selected_metric=metric,
            height=height,
        )
    
    return figures


def create_model_ranking_chart(
    df: pd.DataFrame,
    selected_metric: Literal['wer', 'cer', 'bleu'] = 'wer',
    top_n: Optional[int] = None,
    height: int = 400,
) -> go.Figure:
    """
    Create horizontal bar chart ranking models by overall performance.
    
    Args:
        df: DataFrame with columns ['model', 'dialect', 'wer', 'cer', 'bleu']
        selected_metric: Metric to rank by
        top_n: Number of top models to show. None shows all.
        height: Chart height in pixels
        
    Returns:
        Plotly Figure with horizontal bar chart
    """
    df_filtered = df[df['dialect'] != 'OVERALL'].copy()
    
    # Calculate overall average per model
    model_avg = df_filtered.groupby('model')[selected_metric].mean().reset_index()
    model_avg.columns = ['model', 'avg_metric']
    
    # Sort by metric (ascending for WER/CER, descending for BLEU)
    ascending = selected_metric.lower() != 'bleu'
    model_avg = model_avg.sort_values('avg_metric', ascending=not ascending)
    
    if top_n:
        model_avg = model_avg.head(top_n)
    
    # Get colors for models
    colors = [MODEL_COLORS.get(m, '#7f7f7f') for m in model_avg['model']]
    
    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=model_avg['avg_metric'],
            y=model_avg['model'],
            orientation='h',
            marker_color=colors,
            hovertemplate=(
                "Model: %{y}<br>"
                f"{selected_metric.upper()}: %{{x:.2f}}"
                "<extra></extra>"
            ),
        )
    ])
    
    metric_labels = {
        'wer': 'Average WER (%)',
        'cer': 'Average CER (%)',
        'bleu': 'Average BLEU Score'
    }
    
    fig.update_layout(
        title=dict(
            text=f"Model Ranking by {selected_metric.upper()}",
            x=0.5,
            xanchor='center',
        ),
        xaxis=dict(
            title=metric_labels.get(selected_metric, selected_metric),
        ),
        yaxis=dict(
            title="Model",
            categoryorder='total ascending' if ascending else 'total descending',
        ),
        height=height,
        hovermode='closest',
    )
    
    return fig