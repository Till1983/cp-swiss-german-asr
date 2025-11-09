import streamlit as st
import pandas as pd
from datetime import datetime

def display_data_table(
    df: pd.DataFrame,
    title: str = "Results Table",
    show_pagination: bool = True,
    rows_per_page: int = 50,
    height: int = 600
) -> None:
    """
    Display a formatted and filterable DataFrame using Streamlit's native scrolling.
    
    Args:
        df: DataFrame to display
        title: Title for the table
        show_pagination: Kept for API compatibility (not used - Streamlit handles pagination natively)
        rows_per_page: Kept for API compatibility (not used - Streamlit handles pagination natively)
        height: Height of the dataframe in pixels
    """
    if df.empty:
        st.warning("No data to display")
        return
    
    st.subheader(title)
    
    # Display row count
    total_rows = len(df)
    st.caption(f"Total records: **{total_rows}**")
    
    # Column configuration
    column_config = {
        "dialect": st.column_config.TextColumn(
            "Dialect",
            width="small",
            help="Swiss German dialect code"
        ),
        "wer": st.column_config.NumberColumn(
            "WER (%)",
            width="small",
            format="%.2f",
            help="Word Error Rate - lower is better"
        ),
        "cer": st.column_config.NumberColumn(
            "CER (%)",
            width="small",
            format="%.2f",
            help="Character Error Rate - lower is better"
        ),
        "bleu": st.column_config.NumberColumn(
            "BLEU",
            width="small",
            format="%.2f",
            help="BLEU score - higher is better"
        ),
        "model": st.column_config.TextColumn(
            "Model",
            width="medium",
            help="Model name"
        )
    }
    
    # Display dataframe with native Streamlit scrolling
    # Streamlit automatically handles large datasets efficiently
    st.dataframe(
        df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        height=height  # Creates a scrollable container
    )
    
    # Sorting instructions
    st.info(
        "💡 **Tip:** Click on column headers to sort the table. "
        "Scroll within the table to view more rows."
    )


def display_summary_statistics(df: pd.DataFrame) -> None:
    """
    Display summary statistics for numeric columns.
    
    Args:
        df: DataFrame with results
    """
    st.subheader("Summary Statistics")
    
    # Select only numeric columns (excluding index if present)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns to summarize")
        return
    
    # Calculate statistics
    stats_df = df[numeric_cols].describe().T
    stats_df = stats_df[['mean', 'std', 'min', 'max']]
    
    # Format the statistics
    column_config = {
        "mean": st.column_config.NumberColumn("Mean", format="%.2f"),
        "std": st.column_config.NumberColumn("Std Dev", format="%.2f"),
        "min": st.column_config.NumberColumn("Min", format="%.2f"),
        "max": st.column_config.NumberColumn("Max", format="%.2f")
    }
    
    st.dataframe(
        stats_df,
        column_config=column_config,
        use_container_width=True
    )


def download_filtered_data(df: pd.DataFrame, filename_prefix: str = "filtered_data") -> None:
    """
    Provide a download button for the filtered DataFrame as CSV.
    
    Args:
        df: DataFrame to download
        filename_prefix: Prefix for the downloaded filename
    """
    
    if df.empty:
        st.warning("No data available to download")
        return
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    
    # Convert DataFrame to CSV bytes
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    
    # Display download button
    st.download_button(
        label="📥 Download CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        help="Download the current view as a CSV file"
    )