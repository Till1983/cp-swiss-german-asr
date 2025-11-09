import streamlit as st
from typing import List
import pandas as pd

def filter_dataframe(
    df: pd.DataFrame,
    selected_models: List[str],
    selected_dialects: List[str]
) -> pd.DataFrame:
    """
    Filter DataFrame based on selected models and dialects.
    
    Args:
        df: DataFrame with model and dialect columns
        selected_models: List of model names to include
        selected_dialects: List of dialect codes to include
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    if selected_models:
        filtered_df = filtered_df[filtered_df['model'].isin(selected_models)]
    
    if selected_dialects:
        filtered_df = filtered_df[filtered_df['dialect'].isin(selected_dialects)]
    
    return filtered_df


def render_sidebar(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Render sidebar with filtering and metric selection options.
    
    Args:
        df: DataFrame with results data
        
    Returns:
        Tuple of (filtered_df, selected_metric)
    """
    st.sidebar.header("Filters & Settings")
    
    # Get unique values for filters
    available_models = sorted(df['model'].unique()) if 'model' in df.columns else []
    available_dialects = sorted(df['dialect'].unique())
    
    # Model selection
    selected_models = st.sidebar.multiselect(
        "Select Models",
        options=available_models,
        default=available_models,
        help="Choose which models to display"
    )
    
    # Dialect selection
    selected_dialects = st.sidebar.multiselect(
        "Select Dialects",
        options=available_dialects,
        default=available_dialects,
        help="Choose which Swiss German dialects to display"
    )
    
    # Metric selection
    selected_metric = st.sidebar.radio(
        "Select Metric",
        options=["WER", "CER", "BLEU"],
        index=0,
        help="Choose which metric to visualize"
    )
    
    # Filter the dataframe
    filtered_df = filter_dataframe(df, selected_models, selected_dialects)
    
    # Show filtering information
    st.sidebar.divider()
    st.sidebar.info(
        f"**Filtered Results:**\n\n"
        f"📊 Total samples: {len(filtered_df)}\n\n"
        f"🤖 Models: {len(selected_models)}\n\n"
        f"🗣️ Dialects: {len(selected_dialects)}"
    )
    
    return filtered_df, selected_metric.lower()