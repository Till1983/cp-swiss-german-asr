import streamlit as st
from pathlib import Path
from typing import List  # Added for type hints
import plotly.graph_objects as go
import plotly.io as pio
from utils.data_loader import get_available_results, combine_multiple_models
from components.sidebar import render_sidebar
from components.model_comparison import compare_models, _get_performance_category
from components.dialect_breakdown import create_aggregate_comparison, render_per_dialect_analysis
from components.data_table import display_data_table, display_summary_statistics, download_filtered_data
from components.statistics_panel import render_metrics_definitions
from components.terminology_panel import render_terminology_definitions
from components.plotly_charts import create_wer_by_dialect_chart, create_metric_comparison_chart

# Configure Plotly theme to match Streamlit
pio.templates["streamlit"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Source Sans Pro, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linecolor='rgba(128,128,128,0.4)',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linecolor='rgba(128,128,128,0.4)',
        ),
    )
)
pio.templates.default = "streamlit"

# Page configuration
st.set_page_config(
    page_title="Swiss German ASR Evaluation",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🎙️ Swiss German ASR Model Evaluation Dashboard")
st.markdown("""
This dashboard provides comprehensive evaluation results for Automatic Speech Recognition (ASR) 
models tested on Swiss German dialects. Compare model performance across different dialects 
using metrics like WER (Word Error Rate), CER (Character Error Rate), and BLEU scores.
""")

# Load data
results_dir = Path("results/metrics")

# Get available models
available_models = get_available_results(str(results_dir))

if not available_models:
    st.warning("No evaluation results found. Please run the evaluation script first.")
    st.stop()

# Model selection - Multi-select for comparison
st.sidebar.header("Model Selection")
selected_models = st.sidebar.multiselect(
    "Choose models to compare:",
    options=list(available_models.keys()),
    default=[list(available_models.keys())[0]],  # Default to first model
    help="Select one or more models to analyze and compare"
)

# Show selection summary
if len(selected_models) > 1:
    st.sidebar.success(f"✅ Comparing {len(selected_models)} models")
elif len(selected_models) == 1:
    st.sidebar.info(f"📊 Analyzing: {selected_models[0]}")

# Validate selection
if not selected_models:
    st.warning("⚠️ Please select at least one model to analyze.")
    st.stop()

# Load data for all selected models
try:
    df = combine_multiple_models(selected_models, available_models)
except ValueError as e:
    st.error(f"❌ Failed to load model data: {e}")
    st.stop()

if df.empty:
    st.error("No data available for the selected models.")
    st.stop()

# Apply sidebar filters
filtered_df, selected_metric = render_sidebar(df)

@st.cache_data
def prepare_chart_data(dataframe, metric: str) -> dict:
    """
    Transform DataFrame to format expected by plotly_charts functions.
    
    This function is cached to avoid recomputing the same transformations
    when filters change but data remains the same.
    
    Args:
        dataframe: Filtered DataFrame with model, dialect, and metric columns
        metric: The metric column name (wer, cer, bleu)
        
    Returns:
        Dictionary mapping model names to dictionaries of dialect -> metric values
    """
    chart_data = {}
    
    # Exclude OVERALL rows for dialect charts
    dialect_df = dataframe[dataframe['dialect'] != 'OVERALL']
    
    if 'model' in dialect_df.columns:
        for model_name in dialect_df['model'].unique():
            model_data = dialect_df[dialect_df['model'] == model_name]
            chart_data[model_name] = dict(
                zip(model_data['dialect'], model_data[metric])
            )
    else:
        # Single model case - should not happen with new implementation, but keep as fallback
        chart_data['Unknown Model'] = dict(
            zip(dialect_df['dialect'], dialect_df[metric])
        )
    
    return chart_data


def abbreviate_model_name(model_name: str, max_length: int = 15) -> str:
    """
    Abbreviate model name for file naming.
    
    Args:
        model_name: Full model name
        max_length: Maximum length of abbreviated name
        
    Returns:
        Abbreviated model name
    """
    # Common abbreviations
    abbreviations = {
        'whisper-': 'w-',
        'wav2vec2-': 'w2v2-',
        'large-v3-turbo': 'lv3t',
        'large-v3': 'lv3',
        'large-v2': 'lv2',
        'large': 'lg',
        'medium': 'md',
        'small': 'sm',
        'base': 'bs',
        'tiny': 'tn',
        'german-with-lm': 'de-lm',
        'german': 'de',
    }
    
    abbreviated = model_name
    for full, abbr in abbreviations.items():
        abbreviated = abbreviated.replace(full, abbr)
    
    return abbreviated[:max_length]


def create_download_filename(selected_models: List[str], suffix: str = "overview") -> str:
    """
    Create a clean filename for downloads.
    
    Args:
        selected_models: List of selected model names
        suffix: File suffix (e.g., "overview", "detailed")
        
    Returns:
        Filename string without extension
    """
    if len(selected_models) == 1:
        return f"{selected_models[0]}_{suffix}"
    elif len(selected_models) <= 3:
        abbreviated = "_".join([abbreviate_model_name(m) for m in selected_models])
        return f"{abbreviated}_{suffix}"
    else:
        return f"multi_model_{len(selected_models)}x_{suffix}"


# Tab structure
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🗺️ Dialect Analysis", 
    "📈 Detailed Metrics",
    "🔍 Sample Predictions"
])

with tab1:  # Overview
    st.header("📊 Overview")
    
    # Render metrics definitions contextually
    render_metrics_definitions()
    
    # Display aggregate comparison when comparing multiple models
    if 'model' in filtered_df.columns and filtered_df['model'].nunique() > 1:
        st.subheader("Model Comparison")
        fig_agg = create_aggregate_comparison(
            filtered_df,
            selected_metric=selected_metric
        )
        st.plotly_chart(fig_agg, use_container_width=True)
    
    # Display overall metrics from OVERALL row
    overall_df = filtered_df[filtered_df['dialect'] == 'OVERALL']
    
    col1, col2, col3 = st.columns(3)
    
    if 'wer' in overall_df.columns and not overall_df.empty:
        with col1:
            st.metric("Average WER", f"{overall_df['wer'].mean():.2f}%")
    
    if 'cer' in overall_df.columns and not overall_df.empty:
        with col2:
            st.metric("Average CER", f"{overall_df['cer'].mean():.2f}%")
    
    if 'bleu' in overall_df.columns and not overall_df.empty:
        with col3:
            st.metric("Average BLEU", f"{overall_df['bleu'].mean():.2f}")
    
    # Add summary statistics
    st.divider()
    
    # For summary statistics, exclude OVERALL row to show per-dialect distribution
    dialect_only_df = filtered_df[filtered_df['dialect'] != 'OVERALL']
    display_summary_statistics(dialect_only_df)
    
    # Add download button with improved filename
    download_filename = create_download_filename(selected_models, "overview")
    download_filtered_data(filtered_df, filename_prefix=download_filename)
    
    st.divider()
    st.subheader("Full Results Table")
    display_data_table(
        filtered_df,
        title="Filtered Results",
        show_pagination=True,
        rows_per_page=20,
        height=400
    )

with tab2:  # Dialect Analysis
    st.header("🗺️ Dialect Analysis")
    
    # Show metrics by dialect
    if not filtered_df.empty:
        st.subheader(f"{selected_metric.upper()} by Dialect")
        
        # Check if we have multiple models
        has_model_column = 'model' in filtered_df.columns
        multiple_models = has_model_column and filtered_df['model'].nunique() > 1
        
        # Use plotly_charts for dialect comparison
        if multiple_models:
            st.subheader("Model Comparison Across Dialects")
            
            # Prepare data for plotly_charts
            chart_data = prepare_chart_data(filtered_df, selected_metric)
            
            # Create multi-model comparison chart
            fig_dialect = create_metric_comparison_chart(
                data=chart_data,
                metric_name=selected_metric.upper(),
                title=f"{selected_metric.upper()} by Dialect and Model",
                height=500,
                show_legend=True
            )
            st.plotly_chart(fig_dialect, use_container_width=True)
            
            st.markdown("""
            **Quality Scale:** 
            🟢 Excellent | 🟡 Good | 🔴 Poor
            """)
            
            # Determine sort direction based on metric
            sort_ascending = selected_metric in ['wer', 'cer']  # Lower is better for WER/CER
            
            # Display comparison table
            styled_comparison = compare_models(
                filtered_df, 
                selected_metric=selected_metric,
                sort_ascending=sort_ascending
            )
            st.dataframe(styled_comparison, use_container_width=True)
            st.divider()
        
        # Create color-coded bar chart for selected metric
        st.subheader(f"{selected_metric.upper()} Distribution")
        st.markdown("""
        **Quality Scale:** 
        🟢 Excellent | 🟡 Good | 🔴 Poor
        """)
        
        # Use the plotly_charts module for consistency
        chart_data = prepare_chart_data(filtered_df, selected_metric)

        is_single_model = len(chart_data) == 1
        
        # Create chart using the reusable component
        fig_distribution = create_metric_comparison_chart(
            data=chart_data,
            metric_name=selected_metric.upper(),
            title=f"{selected_metric.upper()} by Dialect",
            height=500,
            show_legend=not is_single_model,
            use_performance_colors=is_single_model
        )
        
        st.plotly_chart(fig_distribution, use_container_width=True)
        
        # Show table with color coding
        st.subheader("Detailed Scores")
        st.markdown("""
        **Quality Scale:** 
        🟢 Excellent | 🟡 Good | 🔴 Poor
        """)
        
        # Apply color formatting to the dialect metrics table
        from components.model_comparison import _apply_color_formatting
        
        dialect_df = filtered_df[filtered_df['dialect'] != 'OVERALL'].groupby('dialect')[selected_metric].mean().reset_index()
        dialect_df.columns = ['Dialect', selected_metric.upper()]
        
        styled_dialect = dialect_df.style.map(
            lambda x: _apply_color_formatting(x, selected_metric) if isinstance(x, (int, float)) else '',
        ).format({selected_metric.upper(): "{:.2f}"})
        
        st.dataframe(styled_dialect, use_container_width=True, hide_index=True)
        
        # ============================================================
        # PER-DIALECT ANALYSIS WITH ERROR BREAKDOWN
        # ============================================================
        st.divider()
        
        # Render the comprehensive per-dialect analysis view
        # Note: For multi-model, we show analysis for the first selected model
        primary_model = selected_models[0] if selected_models else None
        if primary_model:
            if len(selected_models) > 1:
                st.info(f"📊 Showing detailed dialect analysis for: **{primary_model}**")
            render_per_dialect_analysis(
                df=filtered_df,
                error_analysis_dir="results/error_analysis",
                selected_model=primary_model
            )
        
    else:
        st.warning("No data available with current filters.")

with tab3:  # Detailed Metrics
    st.header("📈 Detailed Metrics")
    
    # Add WER by dialect chart
    if not filtered_df.empty and 'wer' in filtered_df.columns:
        st.subheader("WER Comparison by Dialect")
        wer_chart_data = prepare_chart_data(filtered_df, 'wer')
        fig_wer = create_wer_by_dialect_chart(
            data=wer_chart_data,
            title="Word Error Rate by Dialect",
            height=450,
            show_legend=True
        )
        st.plotly_chart(fig_wer, use_container_width=True)
    
    # Add CER comparison if available
    if not filtered_df.empty and 'cer' in filtered_df.columns:
        st.subheader("CER Comparison by Dialect")
        cer_chart_data = prepare_chart_data(filtered_df, 'cer')
        fig_cer = create_metric_comparison_chart(
            data=cer_chart_data,
            metric_name="CER",
            title="Character Error Rate by Dialect",
            height=450,
            show_legend=True
        )
        st.plotly_chart(fig_cer, use_container_width=True)

    # Add BLEU comparison if available
    if not filtered_df.empty and 'bleu' in filtered_df.columns:
        st.subheader("BLEU Score Comparison by Dialect")
        bleu_chart_data = prepare_chart_data(filtered_df, 'bleu')
        fig_bleu = create_metric_comparison_chart(
            data=bleu_chart_data,
            metric_name="BLEU",
            title="BLEU Score by Dialect",
            height=450,
            show_legend=True
        )
        st.plotly_chart(fig_bleu, use_container_width=True)
    
    st.divider()
    
    # Add the detailed data table here
    display_data_table(
        filtered_df,
        title="Complete Metrics Breakdown",
        show_pagination=True,
        rows_per_page=50
    )
    
    # Add download with improved filename
    download_filename = create_download_filename(selected_models, "detailed")
    download_filtered_data(filtered_df, filename_prefix=download_filename)

with tab4:  # Sample Predictions
    st.header("🔍 Error Analysis & Sample Inspection")
    
    # Render terminology definitions for context: Reference, Hypothesis, Word-Level Alignment
    render_terminology_definitions()

    # Import the error sample viewer
    from components.error_sample_viewer import render_worst_samples_viewer
    
    # Show analysis for the first selected model
    primary_model = selected_models[0] if selected_models else None
    if primary_model:
        if len(selected_models) > 1:
            st.info(f"📊 Showing error analysis for: **{primary_model}**")
        
        # Render the worst samples viewer
        render_worst_samples_viewer(
            error_analysis_dir="results/error_analysis",
            selected_model=primary_model
        )