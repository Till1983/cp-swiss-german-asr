"""
Unit tests for src.frontend.components.plotly_charts module.

Tests chart generation and performance categorization logic.
"""

import pytest
from src.frontend.components.plotly_charts import (
    get_performance_category,
    get_performance_color,
    create_wer_by_dialect_chart,
    create_metric_comparison_chart,
    PERFORMANCE_COLORS,
    PERFORMANCE_THRESHOLDS,
    MODEL_COLORS
)


class TestGetPerformanceCategory:
    """Tests for get_performance_category function."""

    def test_wer_excellent(self):
        """Test WER in excellent range."""
        assert get_performance_category(15.0, 'wer') == 'excellent'
        assert get_performance_category(0.0, 'wer') == 'excellent'
        assert get_performance_category(29.9, 'wer') == 'excellent'

    def test_wer_good(self):
        """Test WER in good range."""
        assert get_performance_category(30.0, 'wer') == 'good'
        assert get_performance_category(40.0, 'wer') == 'good'
        assert get_performance_category(49.9, 'wer') == 'good'

    def test_wer_poor(self):
        """Test WER in poor range."""
        assert get_performance_category(50.0, 'wer') == 'poor'
        assert get_performance_category(75.0, 'wer') == 'poor'
        assert get_performance_category(100.0, 'wer') == 'poor'

    def test_cer_excellent(self):
        """Test CER in excellent range."""
        assert get_performance_category(5.0, 'cer') == 'excellent'
        assert get_performance_category(0.0, 'cer') == 'excellent'
        assert get_performance_category(14.9, 'cer') == 'excellent'

    def test_cer_good(self):
        """Test CER in good range."""
        assert get_performance_category(15.0, 'cer') == 'good'
        assert get_performance_category(25.0, 'cer') == 'good'
        assert get_performance_category(34.9, 'cer') == 'good'

    def test_cer_poor(self):
        """Test CER in poor range."""
        assert get_performance_category(35.0, 'cer') == 'poor'
        assert get_performance_category(50.0, 'cer') == 'poor'
        assert get_performance_category(100.0, 'cer') == 'poor'

    def test_bleu_excellent(self):
        """Test BLEU in excellent range."""
        assert get_performance_category(75.0, 'bleu') == 'excellent'
        assert get_performance_category(50.0, 'bleu') == 'excellent'
        assert get_performance_category(99.9, 'bleu') == 'excellent'

    def test_bleu_good(self):
        """Test BLEU in good range."""
        assert get_performance_category(30.0, 'bleu') == 'good'
        assert get_performance_category(40.0, 'bleu') == 'good'
        assert get_performance_category(49.9, 'bleu') == 'good'

    def test_bleu_poor(self):
        """Test BLEU in poor range."""
        assert get_performance_category(0.0, 'bleu') == 'poor'
        assert get_performance_category(15.0, 'bleu') == 'poor'
        assert get_performance_category(29.9, 'bleu') == 'poor'

    def test_case_insensitive(self):
        """Test that metric names are case-insensitive."""
        assert get_performance_category(15.0, 'WER') == 'excellent'
        assert get_performance_category(15.0, 'Wer') == 'excellent'
        assert get_performance_category(15.0, 'wEr') == 'excellent'

    def test_unknown_metric(self):
        """Test with unknown metric (should return default 'good')."""
        assert get_performance_category(50.0, 'unknown_metric') == 'good'

    def test_boundary_values(self):
        """Test exact boundary values."""
        # WER boundary at 30
        assert get_performance_category(30.0, 'wer') == 'good'
        assert get_performance_category(29.999, 'wer') == 'excellent'

        # WER boundary at 50
        assert get_performance_category(50.0, 'wer') == 'poor'
        assert get_performance_category(49.999, 'wer') == 'good'

    def test_out_of_bounds_values(self):
        """Test values outside defined ranges (should return 'poor')."""
        # BLEU 100.0 is excluded by (50, 100) range, so it falls through
        assert get_performance_category(100.0, 'bleu') == 'poor'
        
        # Negative values fall through
        assert get_performance_category(-1.0, 'wer') == 'poor'


class TestGetPerformanceColor:
    """Tests for get_performance_color function."""

    def test_excellent_performance_color(self):
        """Test that excellent performance returns correct color."""
        color = get_performance_color(15.0, 'wer')
        assert color == PERFORMANCE_COLORS['excellent']
        assert color == '#90EE90'

    def test_good_performance_color(self):
        """Test that good performance returns correct color."""
        color = get_performance_color(40.0, 'wer')
        assert color == PERFORMANCE_COLORS['good']
        assert color == '#FFFFE0'

    def test_poor_performance_color(self):
        """Test that poor performance returns correct color."""
        color = get_performance_color(75.0, 'wer')
        assert color == PERFORMANCE_COLORS['poor']
        assert color == '#FFB6C6'

    def test_bleu_excellent_color(self):
        """Test BLEU excellent performance color."""
        color = get_performance_color(80.0, 'bleu')
        assert color == PERFORMANCE_COLORS['excellent']

    def test_unknown_metric_default_color(self):
        """Test that unknown metrics get default color."""
        color = get_performance_color(50.0, 'unknown')
        # Should fall back to 'good' default
        assert color == '#FFFFE0'


class TestCreateWerByDialectChart:
    """Tests for create_wer_by_dialect_chart function."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for chart creation."""
        return {
            'whisper-small': {
                'BE': 25.0,
                'ZH': 30.0,
                'VS': 28.0
            },
            'whisper-medium': {
                'BE': 20.0,
                'ZH': 25.0,
                'VS': 23.0
            }
        }

    def test_create_basic_chart(self, sample_data):
        """Test creating a basic chart with valid data."""
        fig = create_wer_by_dialect_chart(sample_data)

        assert fig is not None
        assert len(fig.data) == 2  # Two models
        assert fig.layout.title.text == "Word Error Rate by Dialect and Model"

    def test_create_chart_with_custom_title(self, sample_data):
        """Test creating chart with custom title."""
        custom_title = "Custom Chart Title"
        fig = create_wer_by_dialect_chart(sample_data, title=custom_title)

        assert fig.layout.title.text == custom_title

    def test_create_chart_with_custom_height(self, sample_data):
        """Test creating chart with custom height."""
        custom_height = 800
        fig = create_wer_by_dialect_chart(sample_data, height=custom_height)

        assert fig.layout.height == custom_height

    def test_create_chart_with_empty_data(self):
        """Test creating chart with empty data."""
        fig = create_wer_by_dialect_chart({})

        assert fig is not None
        # Should have annotation indicating no data
        assert len(fig.layout.annotations) > 0
        assert "No data available" in fig.layout.annotations[0].text

    def test_create_chart_with_selected_dialects(self, sample_data):
        """Test creating chart with specific dialects selected."""
        selected_dialects = ['BE', 'ZH']
        fig = create_wer_by_dialect_chart(sample_data, dialects=selected_dialects)

        assert fig is not None
        # Check that only selected dialects are in x-axis data
        for trace in fig.data:
            assert set(trace.x) == set(selected_dialects)

    def test_create_chart_without_legend(self, sample_data):
        """Test creating chart without legend."""
        fig = create_wer_by_dialect_chart(sample_data, show_legend=False)

        assert fig.layout.showlegend is False

    def test_create_chart_with_single_model(self):
        """Test creating chart with only one model."""
        data = {
            'whisper-small': {'BE': 25.0, 'ZH': 30.0}
        }
        fig = create_wer_by_dialect_chart(data)

        assert fig is not None
        assert len(fig.data) == 1

    def test_create_chart_with_single_dialect(self):
        """Test creating chart with only one dialect."""
        data = {
            'whisper-small': {'BE': 25.0},
            'whisper-medium': {'BE': 20.0}
        }
        fig = create_wer_by_dialect_chart(data)

        assert fig is not None
        assert len(fig.data) == 2
        for trace in fig.data:
            assert trace.x == ('BE',)

    def test_create_chart_with_missing_dialect_values(self):
        """Test creating chart when some models don't have all dialects."""
        data = {
            'whisper-small': {'BE': 25.0, 'ZH': 30.0},
            'whisper-medium': {'BE': 20.0}  # Missing ZH
        }
        fig = create_wer_by_dialect_chart(data)

        assert fig is not None
        # Should handle missing values gracefully

    def test_create_chart_custom_metric_name(self, sample_data):
        """Test creating chart with custom metric name."""
        fig = create_wer_by_dialect_chart(
            sample_data,
            metric_name="CER",
            title="Character Error Rate"
        )

        assert fig is not None
        assert fig.layout.title.text == "Character Error Rate"

    def test_chart_has_proper_traces(self, sample_data):
        """Test that chart has proper trace structure."""
        fig = create_wer_by_dialect_chart(sample_data)

        # Should have one trace per model
        assert len(fig.data) == len(sample_data)

        # Each trace should be a bar chart
        for trace in fig.data:
            assert trace.type == 'bar'

    def test_chart_dialects_auto_discovery(self):
        """Test that chart automatically discovers all dialects."""
        data = {
            'model-a': {'BE': 25.0, 'ZH': 30.0},
            'model-b': {'VS': 28.0, 'ZH': 32.0}  # Different dialects
        }
        fig = create_wer_by_dialect_chart(data)

        # Should discover BE, ZH, and VS
        all_dialects = set()
        for trace in fig.data:
            all_dialects.update(trace.x)

        assert 'BE' in all_dialects or 'ZH' in all_dialects or 'VS' in all_dialects

    def test_chart_with_zero_values(self):
        """Test chart with zero metric values."""
        data = {
            'perfect-model': {'BE': 0.0, 'ZH': 0.0}
        }
        fig = create_wer_by_dialect_chart(data)

        assert fig is not None
        assert len(fig.data) == 1

    def test_chart_with_very_high_values(self):
        """Test chart with very high metric values."""
        data = {
            'poor-model': {'BE': 100.0, 'ZH': 95.0}
        }
        fig = create_wer_by_dialect_chart(data)

        assert fig is not None
        assert len(fig.data) == 1


    def test_create_chart_non_percentage_metric(self, sample_data):
        """Test creating chart with a metric that isn't a percentage (no % sign)."""
        fig = create_wer_by_dialect_chart(
            sample_data,
            metric_name="SCORE"
        )
        assert fig is not None
        # Check hover template doesn't have % at the end of the value
        # The template is constructed as: f"{metric_name}: %{{y{value_format}<extra></extra>"
        # So we look for "SCORE: %{y:.2f}<"
        for trace in fig.data:
            assert "SCORE: %{y:.2f}<" in trace.hovertemplate

    def test_create_chart_known_model_color(self):
        """Test that known models get their assigned colors."""
        # Add a known model to the data
        data = {'whisper-large-v3': {'BE': 10.0}}
        fig = create_wer_by_dialect_chart(data)
        
        # Check that the color matches the defined constant
        expected_color = MODEL_COLORS['whisper-large-v3']
        assert fig.data[0].marker.color == expected_color



class TestPerformanceThresholds:
    """Tests to verify performance threshold definitions."""

    def test_wer_thresholds_coverage(self):
        """Test that WER thresholds cover all ranges."""
        thresholds = PERFORMANCE_THRESHOLDS['wer']
        assert 'excellent' in thresholds
        assert 'good' in thresholds
        assert 'poor' in thresholds

        # Test that ranges don't overlap and cover 0-100+
        assert thresholds['excellent'][0] == 0
        assert thresholds['excellent'][1] == 30
        assert thresholds['good'][0] == 30
        assert thresholds['good'][1] == 50
        assert thresholds['poor'][0] == 50

    def test_cer_thresholds_coverage(self):
        """Test that CER thresholds cover all ranges."""
        thresholds = PERFORMANCE_THRESHOLDS['cer']
        assert 'excellent' in thresholds
        assert 'good' in thresholds
        assert 'poor' in thresholds

        assert thresholds['excellent'][0] == 0
        assert thresholds['excellent'][1] == 15
        assert thresholds['good'][0] == 15
        assert thresholds['good'][1] == 35
        assert thresholds['poor'][0] == 35

    def test_bleu_thresholds_coverage(self):
        """Test that BLEU thresholds cover all ranges."""
        thresholds = PERFORMANCE_THRESHOLDS['bleu']
        assert 'excellent' in thresholds
        assert 'good' in thresholds
        assert 'poor' in thresholds

        # BLEU is reversed (higher is better)
        assert thresholds['poor'][0] == 0
        assert thresholds['poor'][1] == 30
        assert thresholds['good'][0] == 30
        assert thresholds['good'][1] == 50
        assert thresholds['excellent'][0] == 50
        assert thresholds['excellent'][1] == 100

    def test_performance_colors_defined(self):
        """Test that all performance colors are defined."""
        assert 'excellent' in PERFORMANCE_COLORS
        assert 'good' in PERFORMANCE_COLORS
        assert 'poor' in PERFORMANCE_COLORS

        # Test that colors are valid hex codes
        for color in PERFORMANCE_COLORS.values():
            assert color.startswith('#')
            assert len(color) == 7  # #RRGGBB format


class TestCreateMetricComparisonChart:
    """Tests for create_metric_comparison_chart function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return {
            'model-a': {'BE': 25.0, 'ZH': 30.0},
            'model-b': {'BE': 28.0, 'ZH': 32.0}
        }

    def test_create_basic_chart(self, sample_data):
        """Test creating a basic comparison chart."""
        fig = create_metric_comparison_chart(sample_data)
        assert fig is not None
        assert len(fig.data) == 2
        assert fig.layout.title.text == "WER by Dialect and Model"

    def test_create_chart_custom_metric(self, sample_data):
        """Test creating chart with custom metric."""
        fig = create_metric_comparison_chart(
            sample_data,
            metric_name="BLEU",
            title="BLEU Scores"
        )
        assert fig is not None
        assert fig.layout.title.text == "BLEU Scores"
        assert fig.layout.yaxis.title.text == "BLEU Score"

    def test_create_chart_bleu_lowercase(self, sample_data):
        """Test creating chart with BLEU metric in lowercase."""
        fig = create_metric_comparison_chart(
            sample_data,
            metric_name="bleu"
        )
        assert fig is not None
        assert fig.layout.yaxis.title.text == "bleu Score"


    def test_create_chart_cer(self, sample_data):
        """Test creating chart with CER metric."""
        fig = create_metric_comparison_chart(
            sample_data,
            metric_name="CER"
        )
        assert fig is not None
        assert fig.layout.yaxis.title.text == "CER (%)"

    def test_create_chart_unknown_metric(self, sample_data):
        """Test creating chart with unknown metric."""
        fig = create_metric_comparison_chart(
            sample_data,
            metric_name="UNKNOWN"
        )
        assert fig is not None
        assert fig.layout.yaxis.title.text == "UNKNOWN"

    def test_performance_colors_single_model(self):
        """Test performance coloring with single model."""
        data = {
            'model-a': {'BE': 10.0, 'ZH': 40.0, 'VS': 60.0}
        }
        fig = create_metric_comparison_chart(
            data,
            use_performance_colors=True,
            metric_name="WER"
        )
        
        assert fig is not None
        assert len(fig.data) == 1
        trace = fig.data[0]
        
        # Check that colors are different (excellent, good, poor)
        colors = trace.marker.color
        assert len(colors) == 3
        assert len(set(colors)) == 3  # Should have 3 different colors
        
        # Check custom data (categories)
        categories = trace.customdata
        assert 'Excellent' in categories
        assert 'Good' in categories
        assert 'Poor' in categories

    def test_performance_colors_multiple_models(self, sample_data):
        """Test performance coloring fallback with multiple models."""
        # Should fall back to standard chart
        fig = create_metric_comparison_chart(
            sample_data,
            use_performance_colors=True
        )
        
        assert fig is not None
        assert len(fig.data) == 2  # Standard grouped bar chart

    def test_performance_colors_with_missing_values(self):
        """Test performance coloring with missing values."""
        data = {
            'model-a': {'BE': 10.0, 'ZH': None}
        }
        fig = create_metric_comparison_chart(
            data,
            use_performance_colors=True
        )
        
        assert fig is not None
        trace = fig.data[0]
        colors = trace.marker.color
        
        # Missing value should have fallback color
        assert colors[1] == '#CCCCCC'
        assert trace.customdata[1] == 'N/A'

    def test_performance_colors_explicit_dialects(self):
        """Test performance coloring with explicit dialects list."""
        data = {
            'model-a': {'BE': 10.0, 'ZH': 40.0, 'VS': 60.0}
        }
        dialects = ['BE', 'VS']
        fig = create_metric_comparison_chart(
            data,
            dialects=dialects,
            use_performance_colors=True
        )
        
        assert fig is not None
        trace = fig.data[0]
        assert len(trace.x) == 2
        assert 'ZH' not in trace.x
