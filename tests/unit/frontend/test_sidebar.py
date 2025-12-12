"""
Unit tests for src.frontend.components.sidebar module.

Tests filtering and sidebar logic.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.frontend.components.sidebar import filter_dataframe


class TestFilterDataframe:
    """Tests for filter_dataframe function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        return pd.DataFrame({
            'dialect': ['BE', 'ZH', 'VS', 'BE', 'ZH'],
            'wer': [25.0, 30.0, 35.0, 28.0, 32.0],
            'cer': [12.0, 15.0, 18.0, 14.0, 16.0],
            'model': ['model-a'] * 5
        })

    def test_filter_with_selected_dialects(self, sample_df):
        """Test filtering with specific dialects selected."""
        selected_dialects = ['BE', 'ZH']
        filtered = filter_dataframe(sample_df, selected_dialects)

        assert len(filtered) == 4
        assert set(filtered['dialect'].values) == {'BE', 'ZH'}

    def test_filter_with_single_dialect(self, sample_df):
        """Test filtering with a single dialect."""
        selected_dialects = ['BE']
        filtered = filter_dataframe(sample_df, selected_dialects)

        assert len(filtered) == 2
        assert all(filtered['dialect'] == 'BE')

    def test_filter_with_empty_selection(self, sample_df):
        """Test filtering with no dialects selected (should return all)."""
        selected_dialects = []
        filtered = filter_dataframe(sample_df, selected_dialects)

        # Empty selection should return all rows
        assert len(filtered) == len(sample_df)

    def test_filter_with_all_dialects(self, sample_df):
        """Test filtering with all dialects selected."""
        selected_dialects = ['BE', 'ZH', 'VS']
        filtered = filter_dataframe(sample_df, selected_dialects)

        assert len(filtered) == len(sample_df)

    def test_filter_with_nonexistent_dialect(self, sample_df):
        """Test filtering with a dialect that doesn't exist."""
        selected_dialects = ['NONEXISTENT']
        filtered = filter_dataframe(sample_df, selected_dialects)

        assert len(filtered) == 0

    def test_filter_creates_copy(self, sample_df):
        """Test that filtering creates a copy and doesn't modify original."""
        original_len = len(sample_df)
        selected_dialects = ['BE']

        filtered = filter_dataframe(sample_df, selected_dialects)

        # Original should be unchanged
        assert len(sample_df) == original_len
        # Filtered should be different
        assert len(filtered) != original_len

    def test_filter_with_empty_dataframe(self):
        """Test filtering an empty dataframe."""
        empty_df = pd.DataFrame(columns=['dialect', 'wer', 'cer'])
        selected_dialects = ['BE']

        filtered = filter_dataframe(empty_df, selected_dialects)

        assert len(filtered) == 0
        assert list(filtered.columns) == ['dialect', 'wer', 'cer']

    def test_filter_preserves_columns(self, sample_df):
        """Test that filtering preserves all columns."""
        selected_dialects = ['BE']
        filtered = filter_dataframe(sample_df, selected_dialects)

        assert set(filtered.columns) == set(sample_df.columns)

    def test_filter_with_overall_dialect(self, sample_df):
        """Test filtering with OVERALL dialect."""
        # Add OVERALL row
        overall_row = pd.DataFrame({
            'dialect': ['OVERALL'],
            'wer': [30.0],
            'cer': [15.0],
            'model': ['model-a']
        })
        df_with_overall = pd.concat([sample_df, overall_row], ignore_index=True)

        selected_dialects = ['OVERALL', 'BE']
        filtered = filter_dataframe(df_with_overall, selected_dialects)

        assert len(filtered) == 3  # 2 BE + 1 OVERALL
        assert 'OVERALL' in filtered['dialect'].values
        assert 'BE' in filtered['dialect'].values
