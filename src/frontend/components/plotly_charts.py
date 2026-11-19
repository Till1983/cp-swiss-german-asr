"""
Plotly Charts Module for Swiss German ASR Evaluation Dashboard

This module provides visualization functions for ASR evaluation metrics
using Plotly for interactive charts.
"""

from hashlib import sha256
from typing import Dict, Iterable, List, Mapping, Optional
import plotly.graph_objects as go


# Consistent color palette for models
MODEL_COLORS = {
    'whisper-large-v3': '#1f77b4',
    'whisper-large-v2': '#ff7f0e',
    'whisper-large-v2-swiss-german-baseline': '#7b3294',
    'whisper-large-v2-swiss-german-baseline-step1260': '#c2a5cf',
    'whisper-large-v2-swiss-german-ewc-lambda-3000': '#008837',
    'whisper-large-v2-swiss-german-ewc-lambda-3000-step1260': '#a6dba0',
    'whisper-large-v2-swiss-german-ewc-lambda-30000': '#c51b7d',
    'whisper-large-v2-swiss-german-ewc-lambda-30000-step1260': '#f1b6da',
    'whisper-large-v2-swiss-german-ewc-lambda-300000': '#b35806',
    'whisper-large-v2-swiss-german-ewc-lambda-300000-step1260': '#fdb863',
    'whisper-large': '#17becf',
    'whisper-large-v3-turbo': '#aec7e8',
    'whisper-medium': '#2ca02c',
    'whisper-small': '#d62728',
    'whisper-base': '#9467bd',
    'whisper-tiny': '#8c564b',
    'mms-1b': '#e377c2',
    'mms-1b-all': '#f7b6d2',
    'mms-1b-l1107': '#c5b0d5',
    'wav2vec2': '#7f7f7f',
    'wav2vec2-german': '#bcbd22',
    'wav2vec2-german-1b': '#dbdb8d',
    'wav2vec2-german-1b-5gram': '#9edae5',
    'wav2vec2-1b-german-cv11': '#c49c94',
    'wav2vec2-german-300m': '#fbc15e',
    'wav2vec2-ger-nl-adapted': '#8c6d31',
    'wav2vec2-multi-56': '#a55194',
    'wav2vec2-european-1b': '#636363',
    'wav2vec2-european-300m': '#3182bd',
    'wav2vec2-german-with-lm': '#b5bd61',
    'seamless-m4t': '#6b6ecf',
}


def get_model_colors(models: Iterable[str]) -> Dict[str, str]:
    """Allocate distinct colors for an entire model inventory."""
    names = sorted(set(models))
    colors = {name: MODEL_COLORS[name] for name in names if name in MODEL_COLORS}
    used = set(MODEL_COLORS.values())
    for name in names:
        if name in colors:
            continue
        attempt = 0
        while True:
            digest = sha256(f'{name}:{attempt}'.encode('utf-8')).digest()
            color = '#{:02x}{:02x}{:02x}'.format(
                *(48 + channel % 160 for channel in digest[:3])
            )
            if color not in used:
                colors[name] = color
                used.add(color)
                break
            attempt += 1
    return colors


# Performance-based color thresholds
PERFORMANCE_THRESHOLDS = {
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
        'excellent': (50, 101),
        'good': (30, 50),
        'poor': (0, 30)
    },
    'chrf': {
        'excellent': (75, 101),
        'good': (50, 75),
        'poor': (0, 50)
    },
    'semdist': {
        'excellent': (0, 0.2),
        'good': (0.2, 0.4),
        'poor': (0.4, float('inf'))
    }
}

PERFORMANCE_COLORS = {
    'excellent': '#90EE90',  # Light green
    'good': '#FFFFE0',        # Light yellow
    'poor': '#FFB6C6'         # Light red
}


def get_performance_category(value: float, metric: str) -> str:
    """
    Determine performance category based on metric value.
    
    Args:
        value: Metric value
        metric: Metric name ('wer', 'cer', or 'bleu')
        
    Returns:
        Performance category: 'excellent', 'good', or 'poor'
    """
    metric_lower = metric.lower()
    if metric_lower not in PERFORMANCE_THRESHOLDS:
        return 'good'  # Default fallback
    
    thresholds = PERFORMANCE_THRESHOLDS[metric_lower]
    
    for category, (low, high) in thresholds.items():
        if low <= value < high:
            return category
    
    return 'poor'


def get_performance_color(value: float, metric: str) -> str:
    """
    Get color based on performance category.
    
    Args:
        value: Metric value
        metric: Metric name
        
    Returns:
        Hex color code
    """
    category = get_performance_category(value, metric)
    return PERFORMANCE_COLORS.get(category, '#FFFFE0')


def create_wer_by_dialect_chart(
    data: Dict[str, Dict[str, float]],
    dialects: Optional[List[str]] = None,
    title: str = "Word Error Rate by Dialect and Model",
    metric_name: str = "WER",
    height: int = 500,
    show_legend: bool = True,
    model_colors: Optional[Mapping[str, str]] = None,
) -> go.Figure:
    """
    Create a grouped bar chart showing WER by dialect for multiple models.
    
    Args:
        data: Dictionary mapping model names to dictionaries of dialect -> WER values.
              Example: {'whisper-large-v3': {'BE': 15.2, 'BS': 18.5, 'ZH': 12.3}, ...}
        dialects: Optional list of dialect codes to display. If None, uses all dialects
                  found in the data.
        title: Chart title
        metric_name: Name of the metric for hover template (e.g., "WER", "CER", "BLEU")
        height: Chart height in pixels
        show_legend: Whether to show the legend
        model_colors: Optional color mapping allocated from the full model inventory.
        
    Returns:
        Plotly Figure object with grouped bar chart
    """
    if not data:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title, height=height)
        return fig
    
    # Determine dialects to display
    if dialects is None:
        # Collect all unique dialects from all models
        all_dialects = set()
        for model_data in data.values():
            all_dialects.update(model_data.keys())
        dialects = sorted(list(all_dialects))
    
    # Determine hover format based on metric
    if metric_name.upper() in ["WER", "CER"]:
        value_format = ":.2f}%"
    else:
        value_format = ":.2f}"
    
    # Create traces for each model
    traces = []
    if model_colors is None:
        model_colors = get_model_colors(data)
    for model_name, model_data in data.items():
        wer_values = [model_data.get(dialect, None) for dialect in dialects]
        
        trace = go.Bar(
            name=model_name,
            x=dialects,
            y=wer_values,
            marker_color=model_colors[model_name],
            hovertemplate=(
                "Model: %{fullData.name}<br>"
                "Dialect: %{x}<br>"
                f"{metric_name}: %{{y{value_format}<extra></extra>"
            ),
        )
        traces.append(trace)
    
    # Create figure
    fig = go.Figure(data=traces)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        xaxis=dict(
            title="Dialect",
            tickangle=0,
            type='category',
        ),
        yaxis=dict(
            title="WER (%)",
            rangemode='tozero',
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        height=height,
        showlegend=show_legend,
        legend=dict(
            title="Model",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='closest',
    )
    
    return fig


def create_metric_comparison_chart(
    data: Dict[str, Dict[str, float]],
    dialects: Optional[List[str]] = None,
    metric_name: str = "WER",
    title: Optional[str] = None,
    height: int = 500,
    show_legend: bool = True,
    use_performance_colors: bool = False,
    model_colors: Optional[Mapping[str, str]] = None,
) -> go.Figure:
    """
    Create a grouped bar chart for any metric by dialect for multiple models.
    
    This is a generalized version of create_wer_by_dialect_chart that can
    be used for WER, CER, BLEU, or any other metric.
    
    Args:
        data: Dictionary mapping model names to dictionaries of dialect -> metric values.
        dialects: Optional list of dialect codes to display.
        metric_name: Name of the metric for axis label (e.g., "WER", "CER", "BLEU")
        title: Chart title. If None, generates from metric_name.
        height: Chart height in pixels
        show_legend: Whether to show the legend
        use_performance_colors: If True, colors bars by performance (excellent/good/poor)
                                instead of by model. Only works for single-model data.
        model_colors: Optional color mapping allocated from the full model inventory.
        
    Returns:
        Plotly Figure object with grouped bar chart
    """
    if title is None:
        title = f"{metric_name} by Dialect and Model"
    
    # Determine y-axis label based on metric
    if metric_name.upper() in ["WER", "CER"]:
        yaxis_title = f"{metric_name} (%)"
        value_format = ":.2f}"
    elif metric_name.upper() in ["BLEU", "CHRF"]:
        yaxis_title = f"{metric_name} Score"
        value_format = ":.2f}"
    elif metric_name.upper() == "SEMDIST":
        yaxis_title = "Semantic Distance"
        value_format = ":.3f}"
    else:
        yaxis_title = metric_name
        value_format = ":.2f}"
    
    # If performance colors requested and single model, use special coloring
    if use_performance_colors and len(data) == 1:
        model_name = list(data.keys())[0]
        model_data = data[model_name]
        
        if dialects is None:
            dialects = sorted(list(model_data.keys()))
        
        values = [model_data.get(d, None) for d in dialects]
        colors = [get_performance_color(v, metric_name) if v is not None else '#CCCCCC' 
                  for v in values]
        categories = [get_performance_category(v, metric_name).capitalize() if v is not None else 'N/A'
                      for v in values]
        
        fig = go.Figure(data=[
            go.Bar(
                x=dialects,
                y=values,
                marker_color=colors,
                marker_line_color='rgb(8,48,107)',
                marker_line_width=1.5,
                text=[f"{v:.2f}" if v is not None else "N/A" for v in values],
                textposition='outside',
                hovertemplate=(
                    "<b>%{x}</b><br>" +
                    f"{metric_name}: %{{y{value_format}<br>" +
                    "Performance: %{customdata}<br>" +
                    "<extra></extra>"
                ),
                customdata=categories,
            )
        ])
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18)),
            xaxis=dict(title="Dialect", tickangle=0, type='category'),
            yaxis=dict(title=yaxis_title, rangemode='tozero'),
            height=height,
            showlegend=False,
            hovermode='closest',
        )
        
        return fig
    
    # Otherwise, use standard model-colored chart
    fig = create_wer_by_dialect_chart(
        data=data,
        dialects=dialects,
        title=title,
        metric_name=metric_name,  # Pass metric_name to fix hover template
        height=height,
        show_legend=show_legend,
        model_colors=model_colors,
    )
    
    # Update y-axis title
    fig.update_layout(yaxis=dict(title=yaxis_title))
    
    return fig