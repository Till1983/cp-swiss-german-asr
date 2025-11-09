import pandas as pd
from plotly import express as px
from typing import Literal, List

import plotly.graph_objects as go


def create_dialect_comparison(
    df: pd.DataFrame,
    selected_metric: Literal['wer', 'cer', 'bleu'] = 'wer',
    models: List[str] = None
) -> go.Figure:
    """
    Create grouped bar chart comparing models across dialects.
    
    Args:
        df: DataFrame with columns ['model', 'dialect', 'wer', 'cer', 'bleu']
        selected_metric: Metric to display
        models: List of model names to include (None for all)
        
    Returns:
        Plotly figure object
    """
    # Filter out OVERALL rows and optionally filter models
    df_filtered = df[df['dialect'] != 'OVERALL'].copy()
    if models:
        df_filtered = df_filtered[df_filtered['model'].isin(models)]
    
    # Metric display names
    metric_labels = {
        'wer': 'Word Error Rate (%)',
        'cer': 'Character Error Rate (%)',
        'bleu': 'BLEU Score'
    }
    
    # Create grouped bar chart
    fig = px.bar(
        df_filtered,
        x='dialect',
        y=selected_metric,
        color='model',
        barmode='group',
        title=f'{metric_labels[selected_metric]} by Dialect',
        labels={
            'dialect': 'Dialect',
            selected_metric: metric_labels[selected_metric],
            'model': 'Model'
        },
        hover_data={
            'dialect': True,
            'model': True,
            selected_metric: ':.2f'
        }
    )
    
    # Update layout for better readability
    fig.update_layout(
        xaxis_title='Dialect',
        yaxis_title=metric_labels[selected_metric],
        legend_title='Model',
        hovermode='closest'
    )
    
    return fig


def create_aggregate_comparison(
    df: pd.DataFrame,
    selected_metric: Literal['wer', 'cer', 'bleu'] = 'wer',
    models: List[str] = None
) -> go.Figure:
    """
    Create bar chart showing per-model aggregate metrics.
    
    Args:
        df: DataFrame with columns ['model', 'dialect', 'wer', 'cer', 'bleu']
        selected_metric: Metric to display
        models: List of model names to include (None for all)
        
    Returns:
        Plotly figure object
    """
    # Filter to OVERALL rows or calculate aggregates
    if 'OVERALL' in df['dialect'].values:
        df_agg = df[df['dialect'] == 'OVERALL'].copy()
    else:
        df_agg = df.groupby('model')[selected_metric].mean().reset_index()
        df_agg['dialect'] = 'OVERALL'
    
    if models:
        df_agg = df_agg[df_agg['model'].isin(models)]
    
    # Metric display names
    metric_labels = {
        'wer': 'Word Error Rate (%)',
        'cer': 'Character Error Rate (%)',
        'bleu': 'BLEU Score'
    }
    
    # Sort by metric (ascending for WER/CER, descending for BLEU)
    sort_ascending = selected_metric != 'bleu'
    df_agg = df_agg.sort_values(selected_metric, ascending=sort_ascending)
    
    # Create bar chart
    fig = px.bar(
        df_agg,
        x='model',
        y=selected_metric,
        color='model',
        title=f'Overall {metric_labels[selected_metric]} by Model',
        labels={
            'model': 'Model',
            selected_metric: metric_labels[selected_metric]
        },
        hover_data={
            'model': True,
            selected_metric: ':.2f'
        }
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Model',
        yaxis_title=metric_labels[selected_metric],
        showlegend=False,
        hovermode='closest'
    )
    
    return fig