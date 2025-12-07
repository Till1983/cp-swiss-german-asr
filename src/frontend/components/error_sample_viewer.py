"""
Error Sample Viewer Component for Swiss German ASR Dashboard

Provides interactive viewing of worst-performing samples with word-level
alignment visualization and error highlighting.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from utils.error_data_loader import get_worst_samples_path, load_worst_samples


# Color scheme for error types
ERROR_COLORS = {
    'correct': '#90EE90',      # Light green
    'substitution': '#FFB6C6',  # Light red
    'insertion': '#ADD8E6',     # Light blue
    'deletion': '#FFFFE0'       # Light yellow
}


def parse_alignment_visualization(alignment_viz: str) -> List[Dict[str, str]]:
    """
    Parse the alignment visualization string into structured data.
    
    Args:
        alignment_viz: Multi-line alignment string from CSV
        
    Returns:
        List of dicts with 'ref', 'hyp', 'type' for each position
    """
    lines = alignment_viz.strip().split('\n')
    
    if len(lines) != 3:
        return []
    
    # Extract the three lines
    ref_line = lines[0].replace('REF:', '').strip()
    hyp_line = lines[1].replace('HYP:', '').strip()
    type_line = lines[2].replace('TYPE:', '').strip()
    
    # Split by whitespace
    ref_words = ref_line.split()
    hyp_words = hyp_line.split()
    type_markers = type_line.split()
    
    # Ensure all have same length
    max_len = max(len(ref_words), len(hyp_words), len(type_markers))
    
    # Pad if needed
    while len(ref_words) < max_len:
        ref_words.append('*****')
    while len(hyp_words) < max_len:
        hyp_words.append('*****')
    while len(type_markers) < max_len:
        type_markers.append('?')
    
    # Build structured alignment
    alignment = []
    for ref, hyp, typ in zip(ref_words, hyp_words, type_markers):
        error_type = 'correct'
        if typ == 'S':
            error_type = 'substitution'
        elif typ == 'I':
            error_type = 'insertion'
        elif typ == 'D':
            error_type = 'deletion'
        elif typ == 'C':
            error_type = 'correct'
        
        alignment.append({
            'ref': ref,
            'hyp': hyp,
            'type': error_type
        })
    
    return alignment


def render_alignment_comparison(
    reference: str,
    hypothesis: str,
    alignment_viz: str
) -> None:
    """
    Display word-level alignment with color-coded errors.
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text
        alignment_viz: Alignment visualization string
    """
    st.markdown("### Word-Level Alignment")
    
    # Parse alignment
    alignment = parse_alignment_visualization(alignment_viz)
    
    if not alignment:
        st.warning("Could not parse alignment visualization")
        return
    
    # Create HTML for alignment display
    ref_html = "<div style='font-family: monospace; font-size: 14px; margin-bottom: 10px;'>"
    ref_html += "<strong>REF:</strong>&nbsp;&nbsp;"
    
    hyp_html = "<div style='font-family: monospace; font-size: 14px; margin-bottom: 10px;'>"
    hyp_html += "<strong>HYP:</strong>&nbsp;&nbsp;"
    
    type_html = "<div style='font-family: monospace; font-size: 14px;'>"
    type_html += "<strong>TYPE:</strong> "
    
    for item in alignment:
        ref_word = item['ref']
        hyp_word = item['hyp']
        error_type = item['type']
        color = ERROR_COLORS.get(error_type, '#FFFFFF')
        
        # Calculate word width (use max of ref/hyp for alignment)
        width = max(len(ref_word), len(hyp_word), 4) * 8  # 8px per char approx
        
        # Reference word
        if ref_word != '*****':
            ref_html += f"<span style='background-color: {color}; padding: 2px 4px; margin: 0 2px; display: inline-block; min-width: {width}px; text-align: center;'>{ref_word}</span>"
        else:
            ref_html += f"<span style='padding: 2px 4px; margin: 0 2px; display: inline-block; min-width: {width}px; text-align: center; color: #CCCCCC;'>-----</span>"
        
        # Hypothesis word
        if hyp_word != '*****':
            hyp_html += f"<span style='background-color: {color}; padding: 2px 4px; margin: 0 2px; display: inline-block; min-width: {width}px; text-align: center;'>{hyp_word}</span>"
        else:
            hyp_html += f"<span style='padding: 2px 4px; margin: 0 2px; display: inline-block; min-width: {width}px; text-align: center; color: #CCCCCC;'>-----</span>"
        
        # Type marker
        type_marker = {
            'correct': '✓',
            'substitution': '✗',
            'insertion': '+',
            'deletion': '-'
        }.get(error_type, '?')
        
        type_html += f"<span style='padding: 2px 4px; margin: 0 2px; display: inline-block; min-width: {width}px; text-align: center;'>{type_marker}</span>"
    
    ref_html += "</div>"
    hyp_html += "</div>"
    type_html += "</div>"
    
    # Display all three lines
    st.markdown(ref_html, unsafe_allow_html=True)
    st.markdown(hyp_html, unsafe_allow_html=True)
    st.markdown(type_html, unsafe_allow_html=True)
    
    # Legend
    st.markdown("---")
    legend_html = """
    <div style='font-size: 12px; margin-top: 10px;'>
        <strong>Legend:</strong>&nbsp;&nbsp;
        <span style='background-color: #90EE90; padding: 2px 6px; margin: 0 4px;'>✓ Correct</span>
        <span style='background-color: #FFB6C6; padding: 2px 6px; margin: 0 4px;'>✗ Substitution</span>
        <span style='background-color: #ADD8E6; padding: 2px 6px; margin: 0 4px;'>+ Insertion</span>
        <span style='background-color: #FFFFE0; padding: 2px 6px; margin: 0 4px;'>- Deletion</span>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)


def render_sample_navigation(
    total_samples: int,
    current_key: str = "sample_idx"
) -> int:
    """
    Render sample navigation controls.
    
    Args:
        total_samples: Total number of samples available
        current_key: Session state key for current index
        
    Returns:
        Current sample index (0-based)
    """
    # Initialize session state
    if current_key not in st.session_state:
        st.session_state[current_key] = 0
    
    # Ensure index is valid
    if st.session_state[current_key] >= total_samples:
        st.session_state[current_key] = total_samples - 1
    if st.session_state[current_key] < 0:
        st.session_state[current_key] = 0
    
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
    
    with col1:
        if st.button("⬅️ Prev", disabled=(st.session_state[current_key] == 0)):
            st.session_state[current_key] -= 1
            st.rerun()
    
    with col2:
        # Sample selector (1-indexed for display)
        new_idx = st.number_input(
            "Sample",
            min_value=1,
            max_value=total_samples,
            value=st.session_state[current_key] + 1,
            key=f"{current_key}_input"
        ) - 1
        
        if new_idx != st.session_state[current_key]:
            st.session_state[current_key] = new_idx
            st.rerun()
    
    with col3:
        st.markdown(f"**of {total_samples}**")
    
    with col4:
        if st.button("Next ➡️", disabled=(st.session_state[current_key] >= total_samples - 1)):
            st.session_state[current_key] += 1
            st.rerun()
    
    return st.session_state[current_key]


def render_error_sample_card(sample: pd.Series) -> None:
    """
    Display a single error sample with metrics and alignment.
    
    Args:
        sample: Pandas Series with sample data
    """
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("WER", f"{sample['wer']:.2f}%")
    
    with col2:
        st.metric("CER", f"{sample['cer']:.2f}%")
    
    with col3:
        if 'bleu' in sample and pd.notna(sample['bleu']):
            st.metric("BLEU", f"{sample['bleu']:.2f}")
        else:
            st.metric("BLEU", "N/A")
    
    with col4:
        st.metric("Dialect", sample['dialect'])
    
    st.divider()
    
    # Error counts
    st.markdown("**Error Breakdown:**")
    error_col1, error_col2, error_col3 = st.columns(3)
    
    with error_col1:
        st.markdown(f"🔴 **Substitutions:** {sample['substitutions']}")
    
    with error_col2:
        st.markdown(f"🔵 **Insertions:** {sample['insertions']}")
    
    with error_col3:
        st.markdown(f"🟡 **Deletions:** {sample['deletions']}")
    
    st.divider()
    
    # Text comparison
    st.markdown("### Full Text Comparison")
    
    ref_col, hyp_col = st.columns(2)
    
    with ref_col:
        st.markdown("**Reference:**")
        st.info(sample['reference'])
    
    with hyp_col:
        st.markdown("**Hypothesis:**")
        st.warning(sample['hypothesis'])
    
    st.divider()
    
    # Alignment visualization
    render_alignment_comparison(
        sample['reference'],
        sample['hypothesis'],
        sample['alignment_viz']
    )


def render_worst_samples_viewer(
    error_analysis_dir: str = "results/error_analysis",
    selected_model: Optional[str] = None
) -> None:
    """
    Main component for viewing worst-performing samples.
    
    Args:
        error_analysis_dir: Path to error analysis directory
        selected_model: Model name to display samples for
    """
    st.header("🔍 Worst-Performing Samples")
    
    # Check if worst samples exist for this model
    if not selected_model:
        st.warning("No model selected")
        return
    
    csv_path = get_worst_samples_path(selected_model, error_analysis_dir)
    
    if not csv_path:
        st.info(f"""
        **No worst samples data available for {selected_model}.**
        
        To generate error analysis with worst samples, run:
```bash
        docker compose run --rm api python scripts/analyze_errors.py
```
        
        This will create:
        - `analysis_{selected_model}.json` - Full error analysis
        - `worst_samples_{selected_model}.csv` - Worst performing samples
        """)
        return
    
    # Load worst samples
    df = load_worst_samples(csv_path)
    
    if df.empty:
        st.warning(f"Worst samples file is empty: {csv_path}")
        return
    
    st.success(f"Loaded {len(df)} worst-performing samples for **{selected_model}**")
    
    # Filtering options
    st.markdown("---")
    st.subheader("Filters")
    
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        # Dialect filter
        available_dialects = ['All'] + sorted(df['dialect'].unique().tolist())
        selected_dialect = st.selectbox(
            "Filter by Dialect",
            options=available_dialects,
            key="worst_samples_dialect_filter"
        )
    
    with filter_col2:
        # WER threshold filter
        wer_threshold = st.slider(
            "Minimum WER (%)",
            min_value=0.0,
            max_value=float(df['wer'].max()),
            value=0.0,
            step=5.0,
            key="worst_samples_wer_filter"
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_dialect != 'All':
        filtered_df = filtered_df[filtered_df['dialect'] == selected_dialect]
    
    filtered_df = filtered_df[filtered_df['wer'] >= wer_threshold]
    
    if filtered_df.empty:
        st.warning("No samples match the current filters")
        return
    
    # Sort by WER descending (worst first)
    filtered_df = filtered_df.sort_values('wer', ascending=False).reset_index(drop=True)
    
    st.info(f"📊 Showing {len(filtered_df)} samples (filtered from {len(df)} total)")
    
    st.markdown("---")
    
    # Navigation
    current_idx = render_sample_navigation(
        len(filtered_df),
        current_key="worst_samples_current_idx"
    )
    
    st.divider()
    
    # Display current sample
    current_sample = filtered_df.iloc[current_idx]
    render_error_sample_card(current_sample)
    
    # Download option
    st.markdown("---")
    st.download_button(
        label="📥 Download Filtered Samples CSV",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name=f"worst_samples_{selected_model}_filtered.csv",
        mime="text/csv"
    )