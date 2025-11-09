import streamlit as st
import pandas as pd

def display_data_table(
    df: pd.DataFrame,
    title: str = "Results Table",
    show_pagination: bool = True,
    rows_per_page: int = 50,
    height: int = 400
) -> None:
    """
    Display a formatted and filterable DataFrame with pagination.
    
    Args:
        df: DataFrame to display
        title: Title for the table
        show_pagination: Whether to enable pagination
        rows_per_page: Number of rows per page (default: 50)
        height: Height of the dataframe in pixels
    """
    if df.empty:
        st.warning("No data to display")
        return
    
    st.subheader(title)
    
    # Display row count
    total_rows = len(df)
    st.caption(f"Total records: **{total_rows}**")
    
    # Pagination
    if show_pagination and total_rows > rows_per_page:
        # Calculate page range
        max_start = max(0, total_rows - rows_per_page)
        
        start_idx = st.slider(
            "Select row range",
            min_value=0,
            max_value=max_start,
            value=0,
            step=rows_per_page,
            help="Use the slider to navigate through pages of results"
        )
        
        end_idx = min(start_idx + rows_per_page, total_rows)
        df_display = df.iloc[start_idx:end_idx]
        
        st.caption(f"Showing rows {start_idx + 1} to {end_idx} of {total_rows}")
    else:
        df_display = df
    
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
    
    # Display dataframe with formatting
    st.dataframe(
        df_display,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        height=height
    )
    
    # Sorting instructions
    st.info(
        "💡 **Tip:** Click on column headers to sort the table. "
        "Click again to reverse sort order."
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