"""Tests for model comparison tables and charts."""

import pandas as pd
import pytest

from src.frontend.components.model_comparison import (
    compare_models,
    create_model_comparison_chart,
    create_model_ranking_chart,
    create_multi_metric_comparison,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def results():
    return pd.DataFrame([
        {'model': 'model-a', 'dialect': 'BE', 'wer': 10.0, 'bleu': 80.0},
        {'model': 'model-a', 'dialect': 'ZH', 'wer': 30.0, 'bleu': 60.0},
        {'model': 'model-b', 'dialect': 'BE', 'wer': 50.0, 'bleu': 20.0},
        {'model': 'model-b', 'dialect': 'ZH', 'wer': 70.0, 'bleu': 40.0},
        {'model': 'model-a', 'dialect': 'OVERALL', 'wer': 99.0, 'bleu': 0.0},
    ])


def test_comparison_table_averages_dialects_and_formats_performance(results):
    styled = compare_models(results)
    assert styled.data.index.tolist() == ['model-a', 'model-b']
    assert styled.data['OVERALL'].tolist() == [20.0, 60.0]
    html = styled.to_html()
    assert 'background-color: #90EE90' in html
    assert 'background-color: #FFFFE0' in html
    assert 'background-color: #FFB6C6' in html
    assert '20.00' in html

    descending = compare_models(results, sort_ascending=False)
    assert descending.data.index.tolist() == ['model-b', 'model-a']


def test_comparison_chart_filters_overall_and_selected_dialects(results):
    fig = create_model_comparison_chart(
        results, selected_metric='bleu', dialects=['ZH'], height=350,
        show_legend=False,
    )
    assert fig.layout.title.text == 'BLEU Score by Dialect and Model'
    assert fig.layout.height == 350
    assert fig.layout.showlegend is False
    assert {trace.name: list(trace.y) for trace in fig.data} == {
        'model-a': [60.0], 'model-b': [40.0],
    }


def test_multi_metric_charts_use_available_or_explicit_metrics(results):
    figures = create_multi_metric_comparison(results, height=320)
    assert set(figures) == {'wer', 'bleu'}
    assert all(fig.layout.height == 320 for fig in figures.values())
    assert set(create_multi_metric_comparison(results, metrics=['bleu'])) == {'bleu'}


@pytest.mark.parametrize(
    ('metric', 'expected'),
    [('wer', {'model-a': 20.0, 'model-b': 60.0}),
     ('bleu', {'model-a': 70.0, 'model-b': 30.0})],
)
def test_ranking_uses_dialect_averages_not_overall(results, metric, expected):
    trace = create_model_ranking_chart(results, selected_metric=metric).data[0]
    assert dict(zip(trace.y, trace.x)) == expected

