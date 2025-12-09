import streamlit as st
from typing import List
import pandas as pd

def filter_dataframe(
    df: pd.DataFrame,
    selected_dialects: List[str]
) -> pd.DataFrame:
    """
    Filter DataFrame based on selected dialects.
    
    Args:
        df: DataFrame with dialect column
        selected_dialects: List of dialect codes to include
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
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
    available_dialects = sorted(df['dialect'].unique())
    
    # NOTE: Model selection is now handled in app.py main sidebar (multiselect)
    # Removed duplicate "Select Models" filter to avoid confusion
    
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
    
    # Filter the dataframe (only by dialect now, models are filtered in app.py)
    filtered_df = filter_dataframe(df, selected_dialects)
    
    # Show filtering information
    st.sidebar.divider()
    
    # Get model count - handle both single and multi-model cases
    model_count = 1
    if 'model' in df.columns:
        model_count = df['model'].nunique()
    
    st.sidebar.info(
        f"**Filtered Results:**\n\n"
        f"📊 Total samples: {len(filtered_df)}\n\n"
        f"🤖 Models: {model_count}\n\n"
        f"🗣️ Dialects: {len(selected_dialects)}"
    )
    
    return filtered_df, selected_metric.lower()