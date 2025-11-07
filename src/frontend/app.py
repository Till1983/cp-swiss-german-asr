import streamlit as st
from pathlib import Path
from utils.data_loader import load_data
from utils.data_loader import get_available_results
from utils.data_loader import combine_model_results
from components.sidebar import render_sidebar

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
        
        # Create bar chart
        dialect_metrics = filtered_df.groupby('dialect')[selected_metric].mean().sort_values()
        st.bar_chart(dialect_metrics)
        
        # Show table
        st.dataframe(dialect_metrics, use_container_width=True)
    else:
        st.warning("No data available with current filters.")

with tab3:
    st.header("Detailed Metrics")
    
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