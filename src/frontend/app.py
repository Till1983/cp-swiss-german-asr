import streamlit as st
from pathlib import Path
from utils.data_loader import load_data
from utils.data_loader import get_available_results
from utils.data_loader import combine_model_results

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
#if not results_dir.exists():
#    st.error(f"Results directory not found: {results_dir}")
#    st.stop()

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
    df = load_data(str(model_files[0]))
else:
    df = combine_model_results(model_files)

if df.empty:
    st.error("No data available for the selected model.")
    st.stop()

# Tab structure placeholder
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🗺️ Dialect Analysis", 
    "📈 Detailed Metrics",
    "🔍 Sample Predictions"
])

with tab1:
    st.info("Overview tab - Coming soon")
    # TODO: Add overall metrics summary and key insights

with tab2:
    st.info("Dialect Analysis tab - Coming soon")
    # TODO: Add dialect-specific performance comparison

with tab3:
    st.info("Detailed Metrics tab - Coming soon")
    # TODO: Add detailed metric breakdowns and visualizations

with tab4:
    st.info("Sample Predictions tab - Coming soon")
    # TODO: Add sample predictions and error analysis