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
    """Render sidebar with filtering and metric selection options."""
    
    st.sidebar.header("Filters & Settings")
    
    # Get unique values
    available_dialects = sorted(df['dialect'].unique())
    
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
    
    # Filter dataframe
    filtered_df = filter_dataframe(df, selected_dialects)
    
    # Show filtering information
    st.sidebar.divider()
    
    # Calculate counts
    model_count = df['model'].nunique() if 'model' in df.columns else 1
    
    # FIX 1: Separate real dialects from OVERALL
    real_dialects = [d for d in selected_dialects if d != 'OVERALL']
    dialect_count = len(real_dialects)
    
    # FIX 2: Show data dimensions instead of misleading "samples"
    dialect_text = f"🗣️ Dialects: {dialect_count}"
    if 'OVERALL' in selected_dialects:
        dialect_text += " (+ OVERALL)"
    
    st.sidebar.info(
        f"**Filtered Results:**\n\n"
        f"🤖 Models: {model_count}\n\n"
        f"{dialect_text}\n\n"
        f"📊 Data points: {len(filtered_df)}"  # Changed from "Total samples"
    )
    
    return filtered_df, selected_metric.lower()