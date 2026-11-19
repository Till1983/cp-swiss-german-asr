"""Tests for dialect charts and the per-dialect dashboard."""

from importlib import import_module
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest


pytestmark = pytest.mark.unit


@pytest.fixture
def dialect(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / 'src' / 'frontend'))
    return import_module('src.frontend.components.dialect_breakdown')


@pytest.fixture
def results():
    return pd.DataFrame([
        {'model': 'model-a', 'dialect': 'BE', 'wer': 10.0, 'cer': 5.0, 'bleu': 80.0},
        {'model': 'model-a', 'dialect': 'ZH', 'wer': 20.0, 'cer': 10.0, 'bleu': 70.0},
        {'model': 'model-b', 'dialect': 'BE', 'wer': 30.0, 'cer': 15.0, 'bleu': 60.0},
        {'model': 'model-a', 'dialect': 'OVERALL', 'wer': 99.0, 'cer': 99.0, 'bleu': 0.0},
    ])


@pytest.fixture
def ui(dialect, monkeypatch):
    streamlit = MagicMock()
    streamlit.columns.side_effect = lambda count: [MagicMock() for _ in range(count)]
    streamlit.selectbox.return_value = 'BE'
    monkeypatch.setattr(dialect, 'st', streamlit)
    return streamlit


def test_dialect_chart_filters_models_and_aligns_missing_values(dialect, results):
    fig = dialect.create_dialect_comparison(results)
    traces = {trace.name: trace for trace in fig.data}
    assert list(traces['model-a'].x) == ['BE', 'ZH']
    assert list(traces['model-b'].y) == [30.0, None]
    filtered = dialect.create_dialect_comparison(results, models=['model-b'])
    assert [trace.name for trace in filtered.data] == ['model-b']
    assert list(filtered.data[0].x) == ['BE']


def test_aggregate_chart_uses_overall_or_dialect_means_and_filters(dialect, results):
    overall = dialect.create_aggregate_comparison(results, models=['model-a'])
    assert list(overall.data[0].x) == ['model-a']
    assert list(overall.data[0].y) == [99.0]

    no_overall = results[results['dialect'] != 'OVERALL']
    averaged = dialect.create_aggregate_comparison(no_overall, selected_metric='bleu')
    assert list(averaged.data[0].x) == ['model-a', 'model-b']
    assert list(averaged.data[0].y) == [75.0, 60.0]


def test_error_distribution_and_confusion_table(dialect, monkeypatch):
    pie = MagicMock()
    monkeypatch.setattr(dialect.go, 'Pie', pie, raising=False)
    stats = {
        'error_distribution': {'substitution': 4, 'deletion': 0, 'insertion': 2},
        'top_confusions': [[['a', 'b'], 3], [['c', 'd'], 2]],
    }
    data = {'dialect_analysis': {'BE': stats}}
    fig = dialect.create_error_distribution_chart(data, 'BE')
    assert fig is not None
    assert pie.call_args.kwargs['labels'] == ['Substitutions', 'Insertions']
    assert pie.call_args.kwargs['values'] == [4, 2]
    assert dialect.create_error_distribution_chart(data, 'ZH') is None
    assert dialect.create_error_distribution_chart(
        {'dialect_analysis': {'BE': {'error_distribution': {'deletion': 0}}}}, 'BE'
    ) is None
    assert dialect.create_error_distribution_chart({'dialect_analysis': {'BE': {}}}, 'BE') is None

    table = dialect.create_confusion_pairs_table(stats, top_n=1)
    assert table.to_dict('records') == [{'Reference': 'a', 'Hypothesis': 'b', 'Count': 3}]
    assert dialect.create_confusion_pairs_table({}).empty


def test_selector_sorts_options_and_uses_unique_key(dialect, ui):
    assert dialect.render_dialect_selector(['ZH', 'BE'], '_test') == 'BE'
    assert ui.selectbox.call_args.kwargs['options'] == ['BE', 'ZH']
    assert ui.selectbox.call_args.kwargs['key'] == 'dialect_selector__test'


def test_render_complete_analysis_uses_selected_model(dialect, ui, monkeypatch, results):
    pie = MagicMock()
    monkeypatch.setattr(dialect.go, 'Pie', pie, raising=False)
    stats = {
        'sample_count': 12, 'mean_wer': 22.0, 'std_wer': 4.0,
        'mean_cer': 11.0, 'std_cer': 2.0,
        'error_distribution': {'substitution': 3},
        'top_confusions': [[['a', 'b'], 2]],
    }
    analyses = {'other': {}, 'chosen': {'dialect_analysis': {'BE': stats}}}
    load = MagicMock(return_value=analyses)
    monkeypatch.setattr(dialect, 'load_all_error_analyses', load)
    dialect.render_per_dialect_analysis(results, 'analysis-dir', selected_model='chosen')

    load.assert_called_once_with('analysis-dir')
    ui.metric.assert_any_call('WER', '20.00%')
    ui.metric.assert_any_call('CER', '10.00%')
    ui.metric.assert_any_call('BLEU', '70.00')
    ui.metric.assert_any_call('Samples', '12')
    ui.metric.assert_any_call('Mean WER', '22.00%')
    ui.metric.assert_any_call('Mean CER', '11.00%')
    ui.metric.assert_any_call('Total Samples', '12')
    ui.caption.assert_any_call('Analyzing: **chosen**')
    ui.plotly_chart.assert_called_once()
    assert ui.dataframe.call_args.args[0].to_dict('records') == [
        {'Reference': 'a', 'Hypothesis': 'b', 'Count': 2}
    ]


def test_render_handles_missing_data(dialect, ui, monkeypatch, results):
    load = MagicMock(return_value={})
    monkeypatch.setattr(dialect, 'load_all_error_analyses', load)
    dialect.render_per_dialect_analysis(results[results['dialect'] == 'OVERALL'])
    ui.warning.assert_called_with('No dialect data available.')
    load.assert_not_called()

    dialect.render_per_dialect_analysis(results)
    ui.metric.assert_any_call('Samples', 'N/A')
    assert any('Error analysis data not available' in call.args[0] for call in ui.info.call_args_list)

    load.return_value = {'model-a': {'dialect_analysis': {'ZH': {}}}}
    dialect.render_per_dialect_analysis(results, selected_model='not-found')
    ui.caption.assert_any_call('Analyzing: **model-a**')
    ui.warning.assert_called_with('No error analysis data found for BE')


def test_render_handles_empty_error_details(dialect, ui, monkeypatch, results):
    monkeypatch.setattr(
        dialect, 'load_all_error_analyses',
        lambda _: {'model-a': {'dialect_analysis': {'BE': {'mean_wer': 10.0}}}},
    )
    dialect.render_per_dialect_analysis(results)
    ui.info.assert_any_call('No error distribution data available')
    ui.info.assert_any_call('No confusion pairs available')
    ui.metric.assert_any_call('Samples', 'N/A')

