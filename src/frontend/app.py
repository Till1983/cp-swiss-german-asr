import streamlit as st
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd
from utils.data_loader import load_data
from utils.data_loader import get_available_results
from utils.data_loader import combine_model_results
from components.sidebar import render_sidebar
from components.model_comparison import compare_models, _get_performance_category
from components.dialect_breakdown import create_dialect_comparison, create_aggregate_comparison

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

# Tab structure placeholder
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🗺️ Dialect Analysis", 
    "📈 Detailed Metrics",
    "🔍 Sample Predictions"
])

with tab1:
    st.header("Overview")
    
    # Display overall metrics using aggregate comparison
    if 'model' in filtered_df.columns:
        fig_agg = create_aggregate_comparison(
            filtered_df,
            selected_metric=selected_metric
        )
        st.plotly_chart(fig_agg, use_container_width=True)
    
    # Display overall metrics
    col1, col2, col3 = st.columns(3)
    
    if 'wer' in filtered_df.columns:
        with col1:
            st.metric("Average WER", f"{filtered_df['wer'].mean():.2f}%")
    
    if 'cer' in filtered_df.columns:
        with col2:
            st.metric("Average CER", f"{filtered_df['cer'].mean():.2f}%")
    
    if 'bleu' in filtered_df.columns:
        with col3:
            st.metric("Average BLEU", f"{filtered_df['bleu'].mean():.2f}")
    
    # Display filtered data
    st.subheader("Filtered Data")
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.header("Dialect Analysis")
    
    # Show metrics by dialect
    if not filtered_df.empty:
        st.subheader(f"{selected_metric.upper()} by Dialect")
        
        # Check if we have multiple models OR multiple timestamps for the same model
        has_model_column = 'model' in filtered_df.columns
        multiple_models = has_model_column and filtered_df['model'].nunique() > 1
        
        # Use dialect comparison chart if we have model data
        if has_model_column and (multiple_models or len(model_files) > 1):
            st.subheader("Model Comparison Across Dialects")
            fig_dialect = create_dialect_comparison(
                filtered_df,
                selected_metric=selected_metric
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
        
        # Prepare data
        dialect_metrics = filtered_df.groupby('dialect')[selected_metric].mean().reset_index()
        dialect_metrics.columns = ['dialect', 'value']
        dialect_metrics = dialect_metrics.sort_values('value', ascending=True)
        
        # Add performance category and color for each bar
        color_map = {
            'excellent': '#90EE90',  # Light green
            'good': '#FFFFE0',        # Light yellow
            'poor': '#FFB6C6'         # Light red
        }
        
        dialect_metrics['category'] = dialect_metrics['value'].apply(
            lambda x: _get_performance_category(x, selected_metric)
        )
        dialect_metrics['color'] = dialect_metrics['category'].map(color_map)
        
        # Create Plotly bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=dialect_metrics['dialect'],
                y=dialect_metrics['value'],
                marker_color=dialect_metrics['color'],
                marker_line_color='rgb(8,48,107)',
                marker_line_width=1.5,
                text=dialect_metrics['value'].round(2),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>' +
                              selected_metric.upper() + ': %{y:.2f}<br>' +
                              'Performance: %{customdata}<br>' +
                              '<extra></extra>',
                customdata=dialect_metrics['category'].str.capitalize()
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=f"{selected_metric.upper()} by Dialect",
            xaxis_title="Dialect",
            yaxis_title=selected_metric.upper(),
            height=500,
            showlegend=False,
            hovermode='x',
            plot_bgcolor='white',
            xaxis=dict(
                showgrid=False,
                showline=True,
                linecolor='rgb(204, 204, 204)',
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgb(230, 230, 230)',
                showline=True,
                linecolor='rgb(204, 204, 204)',
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show table with color coding
        st.subheader("Detailed Scores")
        st.markdown("""
        **Quality Scale:** 
        🟢 Excellent | 🟡 Good | 🔴 Poor
        """)
        
        # Apply color formatting to the dialect metrics table
        from components.model_comparison import _apply_color_formatting
        
        dialect_df = filtered_df.groupby('dialect')[selected_metric].mean().reset_index()
        dialect_df.columns = ['Dialect', selected_metric.upper()]
        
        styled_dialect = dialect_df.style.map(
            lambda x: _apply_color_formatting(x, selected_metric) if isinstance(x, (int, float)) else '',
        ).format({selected_metric.upper(): "{:.2f}"})
        
        st.dataframe(styled_dialect, use_container_width=True, hide_index=True)
    else:
        st.warning("No data available with current filters.")

with tab3:
    st.header("Detailed Metrics")
    # TODO: Add colour coded quality scale for WER, CER, BLEU across dialects
    if not filtered_df.empty:
        # Show all metrics side by side
        st.subheader("All Metrics Comparison")
        
        metrics_cols = ['wer', 'cer', 'bleu']
        available_metrics = [m for m in metrics_cols if m in filtered_df.columns]
        
        for metric in available_metrics:
            st.subheader(f"{metric.upper()} Scores")
            metric_data = filtered_df.groupby('dialect')[metric].mean()
            st.bar_chart(metric_data)
    else:
        st.warning("No data available with current filters.")

with tab4:
    st.info("Sample Predictions tab - Coming soon")
    # TODO: Add sample predictions and error analysis (requires JSON data)