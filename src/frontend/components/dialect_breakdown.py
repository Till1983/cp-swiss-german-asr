"""
Dialect Breakdown Component for Swiss German ASR Dashboard

Provides visualization and analysis tools for per-dialect model performance,
including error distribution and confusion pattern analysis.

This component handles ONLY visualization and UI logic. Data loading is delegated
to utils.error_data_loader.
"""

import pandas as pd
from typing import Literal, Optional, List, Dict, Any
import plotly.graph_objects as go
import streamlit as st

# Import data loading utilities
from utils.error_data_loader import (
    load_all_error_analyses,
    extract_dialect_statistics,
    extract_confusion_pairs_raw
)


# ============================================================================
# CHART FUNCTIONS - Backward Compatible
# ============================================================================

def create_dialect_comparison(
    df: pd.DataFrame,
    selected_metric: Literal['wer', 'cer', 'bleu'] = 'wer',
    models: Optional[List[str]] = None
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
    
    # Get unique models and dialects
    unique_models = df_filtered['model'].unique()
    unique_dialects = sorted(df_filtered['dialect'].unique())
    
    # Create traces for each model
    traces = []
    for model in unique_models:
        model_data = df_filtered[df_filtered['model'] == model]
        
        # Create dictionary mapping dialect to metric value
        dialect_values = {row['dialect']: row[selected_metric] 
                         for _, row in model_data.iterrows()}
        
        # Get values in dialect order (None if missing)
        y_values = [dialect_values.get(d) for d in unique_dialects]
        
        trace = go.Bar(
            name=model,
            x=unique_dialects,
            y=y_values,
            hovertemplate=(
                f"<b>{model}</b><br>" +
                "Dialect: %{x}<br>" +
                f"{metric_labels[selected_metric]}: %{{y:.2f}}<br>" +
                "<extra></extra>"
            )
        )
        traces.append(trace)
    
    # Create figure
    fig = go.Figure(data=traces)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{metric_labels[selected_metric]} by Dialect',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Dialect',
        yaxis_title=metric_labels[selected_metric],
        barmode='group',
        hovermode='closest',
        legend_title='Model',
        height=500
    )
    
    return fig


def create_aggregate_comparison(
    df: pd.DataFrame,
    selected_metric: Literal['wer', 'cer', 'bleu'] = 'wer',
    models: Optional[List[str]] = None
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
    fig = go.Figure(data=[
        go.Bar(
            x=df_agg['model'],
            y=df_agg[selected_metric],
            marker_color='#1f77b4',
            hovertemplate=(
                "Model: %{x}<br>" +
                f"{metric_labels[selected_metric]}: %{{y:.2f}}<br>" +
                "<extra></extra>"
            )
        )
    ])
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Overall {metric_labels[selected_metric]} by Model',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Model',
        yaxis_title=metric_labels[selected_metric],
        showlegend=False,
        hovermode='closest',
        height=400
    )
    
    return fig


# ============================================================================
# DAY 6 VISUALIZATION FUNCTIONS
# ============================================================================

def create_error_distribution_chart(
    error_data: Dict[str, Any],
    dialect: str
) -> Optional[go.Figure]:
    """
    Create pie chart showing error type distribution for a specific dialect.
    
    Args:
        error_data: Error analysis data for a model
        dialect: Dialect code (e.g., 'BE', 'ZH')
        
    Returns:
        Plotly figure object or None if no data
    """
    # Get dialect statistics using data loader
    dialect_stats = extract_dialect_statistics(error_data, dialect)
    
    if not dialect_stats or 'error_distribution' not in dialect_stats:
        return None
    
    error_dist = dialect_stats['error_distribution']
    
    # Extract error counts (not rates)
    labels = []
    values = []
    colors = []
    
    error_types = {
        'substitution': ('#e74c3c', 'Substitutions'),
        'deletion': ('#f39c12', 'Deletions'),
        'insertion': ('#3498db', 'Insertions')
    }
    
    for error_type, (color, label) in error_types.items():
        if error_type in error_dist and error_dist[error_type] > 0:
            labels.append(label)
            values.append(error_dist[error_type])
            colors.append(color)
    
    if not values:
        return None
    
    # Create pie chart
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hovertemplate=(
                "<b>%{label}</b><br>" +
                "Count: %{value}<br>" +
                "Percentage: %{percent}<br>" +
                "<extra></extra>"
            )
        )
    ])
    
    fig.update_layout(
        title=dict(
            text=f"Error Distribution for {dialect}",
            x=0.5,
            xanchor='center'
        ),
        height=400,
        showlegend=True
    )
    
    return fig


def create_confusion_pairs_table(
    dialect_stats: Dict[str, Any],
    top_n: int = 10
) -> pd.DataFrame:
    """
    Create DataFrame of top confusion pairs for display.
    
    Args:
        dialect_stats: Statistics for specific dialect
        top_n: Number of top pairs to show
        
    Returns:
        DataFrame with columns ['Reference', 'Hypothesis', 'Count']
    """
    # Get raw confusion pairs using data loader
    confusion_tuples = extract_confusion_pairs_raw(dialect_stats, top_n)
    
    if not confusion_tuples:
        return pd.DataFrame(columns=['Reference', 'Hypothesis', 'Count'])
    
    # Convert to DataFrame
    rows = [
        {'Reference': ref, 'Hypothesis': hyp, 'Count': count}
        for ref, hyp, count in confusion_tuples
    ]
    
    return pd.DataFrame(rows)


# ============================================================================
# DAY 6 MAIN COMPONENT
# ============================================================================

def render_dialect_selector(
    available_dialects: List[str],
    key_suffix: str = ""
) -> str:
    """
    Render Streamlit dialect selector widget.
    
    Args:
        available_dialects: List of dialect codes
        key_suffix: Unique suffix for widget key (for multiple instances)
        
    Returns:
        Selected dialect code
    """
    selected = st.selectbox(
        "Select Dialect",
        options=sorted(available_dialects),
        key=f"dialect_selector_{key_suffix}",
        help="Choose a Swiss German dialect to analyze"
    )
    
    return selected


def render_per_dialect_analysis(
    df: pd.DataFrame,
    error_analysis_dir: str = "results/error_analysis",
    selected_model: Optional[str] = None
) -> None:
    """
    Render complete per-dialect analysis view with metrics, error distribution,
    and confusion pairs.
    
    This is the main Day 6 component that should be called from app.py.
    
    Args:
        df: DataFrame with evaluation metrics
        error_analysis_dir: Path to error analysis directory
        selected_model: Model name for error analysis (None for first available)
    """
    st.header("Per-Dialect Analysis")
    
    # Get available dialects from DataFrame
    available_dialects = sorted([
        d for d in df['dialect'].unique() 
        if d != 'OVERALL'
    ])
    
    if not available_dialects:
        st.warning("No dialect data available.")
        return
    
    # Dialect selector
    selected_dialect = render_dialect_selector(available_dialects, key_suffix="main")
    
    st.divider()
    
    # === Performance Metrics Section ===
    st.subheader(f"📊 Performance Metrics - {selected_dialect}")
    
    # Get metrics for selected dialect from main DataFrame
    dialect_data = df[df['dialect'] == selected_dialect]
    
    if dialect_data.empty:
        st.warning(f"No data available for {selected_dialect}")
        return
    
    # Load error analysis data using data loader utility
    error_data_all = load_all_error_analyses(error_analysis_dir)
    
    # Determine which model to display
    if selected_model and selected_model in error_data_all:
        error_data = error_data_all[selected_model]
        model_display = selected_model
    elif error_data_all:
        model_display = list(error_data_all.keys())[0]
        error_data = error_data_all[model_display]
    else:
        error_data = None
        model_display = None
    
    # Get dialect statistics and true sample count
    dialect_stats = None
    true_sample_count = None
    if error_data:
        dialect_stats = extract_dialect_statistics(error_data, selected_dialect)
        if dialect_stats and 'sample_count' in dialect_stats:
            true_sample_count = dialect_stats['sample_count']
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'wer' in dialect_data.columns:
            avg_wer = dialect_data['wer'].mean()
            st.metric("WER", f"{avg_wer:.2f}%")
    
    with col2:
        if 'cer' in dialect_data.columns:
            avg_cer = dialect_data['cer'].mean()
            st.metric("CER", f"{avg_cer:.2f}%")
    
    with col3:
        if 'bleu' in dialect_data.columns:
            avg_bleu = dialect_data['bleu'].mean()
            st.metric("BLEU", f"{avg_bleu:.2f}")
    
    with col4:
        # Display true sample count from error analysis
        if true_sample_count is not None:
            st.metric("Samples", f"{true_sample_count}")
        else:
            st.metric("Samples", "N/A")
            st.caption("⚠️ Run error analysis to see sample count")
    
    st.divider()
    
    # === Error Analysis Section ===
    st.subheader("🔍 Error Analysis")
    
    if not error_data:
        st.info("""
        **Error analysis data not available.**
        
        To generate error analysis, run:
```bash
        docker compose run --rm api python scripts/analyze_errors.py
```
        """)
        return
    
    st.caption(f"Analyzing: **{model_display}**")
    
    if not dialect_stats:
        st.warning(f"No error analysis data found for {selected_dialect}")
        return
    
    # Two-column layout for error distribution and confusion pairs
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### Error Type Distribution")
        
        # Create error distribution chart
        error_chart = create_error_distribution_chart(error_data, selected_dialect)
        
        if error_chart:
            st.plotly_chart(error_chart, use_container_width=True)
        else:
            st.info("No error distribution data available")
    
    with col_right:
        st.markdown("### Top Confusion Pairs")
        
        # Create confusion pairs table
        confusion_df = create_confusion_pairs_table(dialect_stats, top_n=10)
        
        if not confusion_df.empty:
            st.dataframe(
                confusion_df,
                use_container_width=True,
                hide_index=True,
                height=400
            )
        else:
            st.info("No confusion pairs available")
    
    # Additional statistics
    st.divider()
    st.subheader("📈 Statistical Summary")
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    
    with stat_col1:
        if 'mean_wer' in dialect_stats:
            st.metric("Mean WER", f"{dialect_stats['mean_wer']:.2f}%")
        if 'std_wer' in dialect_stats:
            st.caption(f"Std Dev: {dialect_stats['std_wer']:.2f}")
    
    with stat_col2:
        if 'mean_cer' in dialect_stats:
            st.metric("Mean CER", f"{dialect_stats['mean_cer']:.2f}%")
        if 'std_cer' in dialect_stats:
            st.caption(f"Std Dev: {dialect_stats['std_cer']:.2f}")
    
    with stat_col3:
        if 'sample_count' in dialect_stats:
            st.metric("Total Samples", f"{dialect_stats['sample_count']}")