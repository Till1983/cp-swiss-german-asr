import streamlit as st
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio
from utils.data_loader import load_data
from utils.data_loader import get_available_results
from utils.data_loader import combine_model_results
from components.sidebar import render_sidebar
from components.model_comparison import compare_models, _get_performance_category
from components.dialect_breakdown import create_dialect_comparison, create_aggregate_comparison, render_per_dialect_analysis
from components.data_table import display_data_table, display_summary_statistics, download_filtered_data
from components.statistics_panel import render_metrics_definitions
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

# Model selection
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a model to analyze:",
    options=list(available_models.keys()),
    index=0
)

# Load selected model data
model_files = available_models[selected_model]
if len(model_files) == 1:
    df = load_data(model_files[0]['csv_path'])  # Access csv_path from dict
else:
    df = combine_model_results(model_files)

if df.empty:
    st.error("No data available for the selected model.")
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
        # Single model case - use selected_model as key
        chart_data[selected_model] = dict(
            zip(dialect_df['dialect'], dialect_df[metric])
        )
    
    return chart_data

# Tab structure placeholder
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🗺️ Dialect Analysis", 
    "📈 Detailed Metrics",
    "🔍 Sample Predictions"
])

with tab1:  # Overview
    st.header("Overview")
    
    # Render metrics definitions contextually
    render_metrics_definitions()
    
    # Display aggregate comparison only when comparing multiple models
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
    
    # Add download button
    download_filtered_data(filtered_df, filename_prefix=f"{selected_model}_overview")
    
    st.divider()
    st.subheader("Full Results Table")
    display_data_table(
        filtered_df,
        title="Filtered Results",
        show_pagination=True,
        rows_per_page=20,
        height=400
    )

with tab2:
    st.header("Dialect Analysis")
    
    # Show metrics by dialect
    if not filtered_df.empty:
        st.subheader(f"{selected_metric.upper()} by Dialect")
        
        # Check if we have multiple models OR multiple timestamps for the same model
        has_model_column = 'model' in filtered_df.columns
        multiple_models = has_model_column and filtered_df['model'].nunique() > 1
        
        # Use new plotly_charts for dialect comparison
        if has_model_column and (multiple_models or len(model_files) > 1):
            st.subheader("Model Comparison Across Dialects")
            
            # Prepare data for plotly_charts
            chart_data = prepare_chart_data(filtered_df, selected_metric)
            
            # Use new create_metric_comparison_chart
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
        
        # Create color-coded bar chart for selected metric using new component
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
        # DAY 6: PER-DIALECT ANALYSIS WITH ERROR BREAKDOWN
        # ============================================================
        st.divider()
        
        # Render the comprehensive per-dialect analysis view
        render_per_dialect_analysis(
            df=filtered_df,
            error_analysis_dir="results/error_analysis",
            selected_model=selected_model
        )
        
    else:
        st.warning("No data available with current filters.")

with tab3:  # Detailed Metrics
    st.header("Detailed Metrics")
    
    # Add WER by dialect chart using new component
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
    
    # Add download
    download_filtered_data(filtered_df, filename_prefix=f"{selected_model}_detailed")

with tab4:
    st.header("🔍 Error Analysis & Sample Inspection")
    
    st.info("""
    **Coming in Day 7**: This tab will provide detailed error analysis including:
    
    📊 **Features to be implemented:**
    - **Worst-performing samples** with aligned reference/hypothesis comparison
    - **Color-coded error highlighting** (substitutions, deletions, insertions)
    - **Per-dialect confusion patterns** showing common word substitution errors
    - **Interactive sample navigation** with prev/next buttons
    - **Statistical error distribution** breakdown by error type
    
    ⚙️ **Requirements:**
    - Error analysis outputs from `scripts/analyze_errors.py`
    - Files: `analysis_*.json` and `worst_samples_*.csv`
    
    💡 **Tip:** Run error analysis with:
```bash
    docker compose run --rm api python scripts/analyze_errors.py
```
    """)
    
    # Show placeholder for future implementation
    st.markdown("---")
    st.markdown("### Preview: Sample Error Alignment")
    st.code("""
    REF:  das    ist    ein    test
    HYP:  das    isch   ei     test   extra
    TYPE: ✓      ✗      ✗      ✓      +
    
    Legend: ✓ = Correct | ✗ = Substitution | - = Deletion | + = Insertion
    """, language="text")