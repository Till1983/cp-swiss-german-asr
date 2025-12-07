"""Unit tests for data loader module."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.data.loader import load_swiss_german_metadata, load_audio


class TestLoadSwissGermanMetadata:
    """Test suite for load_swiss_german_metadata function."""

    @pytest.mark.unit
    def test_loads_valid_tsv(self, mock_swiss_german_data):
        """Test loading valid TSV file."""
        df = load_swiss_german_metadata(str(mock_swiss_german_data))

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'sentence' in df.columns
        assert 'accent' in df.columns

    @pytest.mark.unit
    def test_returns_dataframe_with_expected_columns(self, mock_swiss_german_data):
        """Test returned DataFrame has expected columns."""
        df = load_swiss_german_metadata(str(mock_swiss_german_data))

        expected_columns = {'client_id', 'path', 'sentence', 'accent'}
        assert expected_columns.issubset(set(df.columns))

    @pytest.mark.unit
    def test_raises_on_nonexistent_file(self):
        """Test raises error for non-existent file."""
        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
            load_swiss_german_metadata("nonexistent_file.tsv")

    @pytest.mark.unit
    def test_handles_empty_tsv(self, temp_dir):
        """Test handling of TSV file with only headers."""
        empty_tsv = temp_dir / "empty.tsv"
        empty_tsv.write_text("client_id\tpath\tsentence\n")

        df = load_swiss_german_metadata(str(empty_tsv))
        assert len(df) == 0
        assert list(df.columns) == ['client_id', 'path', 'sentence']

    @pytest.mark.unit
    def test_loads_all_rows(self, mock_swiss_german_data):
        """Test all rows are loaded from TSV."""
        df = load_swiss_german_metadata(str(mock_swiss_german_data))

        # Our mock file has 3 data rows
        assert len(df) == 3

    @pytest.mark.unit
    def test_preserves_unicode_characters(self, temp_dir):
        """Test Swiss German umlauts and special characters are preserved."""
        tsv_content = "client_id\tpath\tsentence\taccent\nclient_001\ttest.wav\tGrueezi Zuerich aeoe\tZH\n"
        tsv_file = temp_dir / "unicode.tsv"
        tsv_file.write_text(tsv_content, encoding='utf-8')

        df = load_swiss_german_metadata(str(tsv_file))

        assert 'Grueezi Zuerich aeoe' in df['sentence'].values


class TestLoadAudio:
    """Test suite for load_audio function."""

    @pytest.mark.unit
    def test_loads_valid_audio_file(self, sample_audio_path):
        """Test loading valid audio file."""
        audio = load_audio(str(sample_audio_path), sample_rate=16000)

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32 or audio.dtype == np.float64
        assert len(audio) > 0

    @pytest.mark.unit
    def test_returns_correct_dtype(self, sample_audio_path):
        """Test audio is returned as float array."""
        audio = load_audio(str(sample_audio_path), sample_rate=16000)

        assert np.issubdtype(audio.dtype, np.floating)

    @pytest.mark.unit
    @patch('librosa.load')
    def test_resamples_to_target_rate(self, mock_load):
        """Test audio is resampled to target sample rate."""
        mock_load.return_value = (np.random.randn(16000).astype(np.float32), 16000)

        audio = load_audio("dummy.wav", sample_rate=16000)

        mock_load.assert_called_once_with("dummy.wav", sr=16000)
        assert isinstance(audio, np.ndarray)

    @pytest.mark.unit
    @patch('librosa.load')
    def test_custom_sample_rate(self, mock_load):
        """Test loading audio with custom sample rate."""
        mock_load.return_value = (np.random.randn(8000).astype(np.float32), 8000)

        audio = load_audio("dummy.wav", sample_rate=8000)

        mock_load.assert_called_once_with("dummy.wav", sr=8000)

    @pytest.mark.unit
    def test_raises_on_nonexistent_file(self):
        """Test raises RuntimeError for non-existent audio file."""
        with pytest.raises(RuntimeError, match="Error loading audio file"):
            load_audio("nonexistent_audio.wav")

    @pytest.mark.unit
    @patch('librosa.load', side_effect=Exception("Corrupt file"))
    def test_handles_corrupt_audio(self, mock_load):
        """Test handling of corrupt audio files."""
        with pytest.raises(RuntimeError, match="Error loading audio file"):
            load_audio("corrupt.wav")

    @pytest.mark.unit
    def test_audio_length_matches_expected(self, sample_audio_path):
        """Test audio length is approximately correct for sample rate."""
        audio = load_audio(str(sample_audio_path), sample_rate=16000)

        # Our sample audio is 1 second at 16kHz = ~16000 samples
        assert 15000 <= len(audio) <= 17000

    @pytest.mark.unit
    @patch('librosa.load')
    def test_handles_different_extensions(self, mock_load):
        """Test loading different audio file extensions."""
        mock_load.return_value = (np.random.randn(16000).astype(np.float32), 16000)

        for ext in ['.wav', '.mp3', '.flac']:
            audio = load_audio(f"test{ext}", sample_rate=16000)
            assert isinstance(audio, np.ndarray)
