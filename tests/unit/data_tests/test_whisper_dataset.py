"""Tests for Whisper dataset metadata and dialect alignment."""

import pandas as pd
import pytest

from src.data.whisper_dataset import WhisperSpeechDataset, load_metadata_df


pytestmark = pytest.mark.unit


def test_metadata_subsampling_is_reproducible_and_aligned(tmp_path):
    source = pd.DataFrame({
        'path': [f'{i}.wav' for i in range(10)],
        'sentence': [f'text {i}' for i in range(10)],
        'accent': [f'dialect {i}' for i in range(10)],
    })
    path = tmp_path / 'metadata.tsv'
    source.to_csv(path, sep='\t', index=False)
    sample = load_metadata_df(path, subset_size=4, seed=7)
    pd.testing.assert_frame_equal(sample, load_metadata_df(path, subset_size=4, seed=7))
    assert len(sample) == 4
    dataset = WhisperSpeechDataset(sample, tmp_path, processor=None)
    assert len(dataset) == 4
    assert dataset.references == sample['sentence'].tolist()
    assert dataset.dialects == sample['accent'].tolist()
    assert len(load_metadata_df(path, subset_size=20)) == 10


def test_dataset_uses_unknown_when_dialect_column_missing(tmp_path):
    df = pd.DataFrame({'path': ['a.wav', 'b.wav'], 'sentence': ['a', 'b']})
    dataset = WhisperSpeechDataset(df, tmp_path, processor=None)
    assert dataset.dialects == ['unknown', 'unknown']

