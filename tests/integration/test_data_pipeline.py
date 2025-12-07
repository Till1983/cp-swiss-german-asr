"""Integration tests for data loading pipeline."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.loader import load_swiss_german_metadata, load_audio
from src.data.preprocessor import AudioPreprocessor


class TestDataPipelineIntegration:
    """Test complete data pipeline integration."""

    @pytest.mark.integration
    def test_load_metadata_and_validate_columns(self, mock_swiss_german_data):
        """Test loading metadata and validating column structure."""
        df = load_swiss_german_metadata(str(mock_swiss_german_data))

        assert len(df) > 0
        required_cols = {'path', 'sentence', 'accent'}
        assert required_cols.issubset(set(df.columns))

    @pytest.mark.integration
    def test_load_audio_from_metadata(self, mock_swiss_german_data, fixtures_root):
        """Test loading audio files referenced in metadata."""
        df = load_swiss_german_metadata(str(mock_swiss_german_data))

        # Load first audio file
        first_audio_path = fixtures_root / "audio" / df.iloc[0]['path']
        if first_audio_path.exists():
            audio = load_audio(str(first_audio_path))

            assert isinstance(audio, np.ndarray)
            assert len(audio) > 0

    @pytest.mark.integration
    def test_load_and_preprocess_audio(self, sample_audio_path):
        """Test loading and preprocessing audio end-to-end."""
        # Load audio
        audio = load_audio(str(sample_audio_path), sample_rate=16000)

        # Preprocess
        preprocessor = AudioPreprocessor(target_sample_rate=16000)
        processed, sr = preprocessor.preprocess(audio, 16000)

        assert sr == 16000
        assert len(processed) == len(audio)
        assert np.abs(processed.mean()) < 0.1  # Normalized

    @pytest.mark.integration
    def test_load_preprocess_different_sample_rate(self, sample_audio_path):
        """Test loading audio and preprocessing to different sample rate."""
        # Load audio at 16kHz
        audio = load_audio(str(sample_audio_path), sample_rate=16000)

        # Preprocess to 8kHz (downsample)
        preprocessor = AudioPreprocessor(target_sample_rate=8000)
        processed, sr = preprocessor.preprocess(audio, 16000)

        assert sr == 8000
        # Should be approximately half the samples
        assert 0.4 < len(processed) / len(audio) < 0.6

    @pytest.mark.integration
    def test_multiple_audio_files_pipeline(self, mock_swiss_german_data, fixtures_root):
        """Test processing multiple audio files from metadata."""
        df = load_swiss_german_metadata(str(mock_swiss_german_data))
        preprocessor = AudioPreprocessor(target_sample_rate=16000)

        processed_count = 0
        for _, row in df.iterrows():
            audio_path = fixtures_root / "audio" / row['path']
            if audio_path.exists():
                audio = load_audio(str(audio_path), sample_rate=16000)
                processed, sr = preprocessor.preprocess(audio, 16000)
                processed_count += 1

                assert sr == 16000
                assert len(processed) > 0

        assert processed_count > 0

    @pytest.mark.integration
    def test_data_pipeline_with_validation(self, mock_swiss_german_data, fixtures_root):
        """Test full pipeline with data validation."""
        from src.utils.audio_utils import validate_audio_file, get_audio_duration

        df = load_swiss_german_metadata(str(mock_swiss_german_data))

        valid_files = 0
        for _, row in df.iterrows():
            audio_path = str(fixtures_root / "audio" / row['path'])

            if validate_audio_file(audio_path):
                duration = get_audio_duration(audio_path)

                assert duration is not None
                assert duration > 0
                valid_files += 1

        assert valid_files > 0


class TestDataSplitPipeline:
    """Test data splitting pipeline."""

    @pytest.mark.integration
    def test_create_and_validate_splits(self, temp_dir):
        """Test creating splits and validating output."""
        from src.data.splitter import create_splits

        # Create input TSV with enough samples
        input_tsv = temp_dir / "input.tsv"
        content = "client_id\tpath\tsentence\taccent\n"
        for i in range(30):
            dialect = ["BE", "ZH", "VS"][i % 3]
            content += f"client_{i}\taudio_{i}.wav\tSentence {i}\t{dialect}\n"
        input_tsv.write_text(content)

        output_dir = temp_dir / "splits"
        output_dir.mkdir()

        result = create_splits(str(input_tsv), str(output_dir), data_root=str(temp_dir))

        # Verify all splits were created
        assert (output_dir / "train.tsv").exists()
        assert (output_dir / "val.tsv").exists()
        assert (output_dir / "test.tsv").exists()

        # Verify split counts
        assert result['train'] + result['val'] + result['test'] == 30

    @pytest.mark.integration
    def test_splits_are_loadable(self, temp_dir):
        """Test that created splits can be loaded as metadata."""
        from src.data.splitter import create_splits

        # Create input TSV
        input_tsv = temp_dir / "input.tsv"
        content = "client_id\tpath\tsentence\taccent\n"
        for i in range(30):
            dialect = ["BE", "ZH", "VS"][i % 3]
            content += f"client_{i}\taudio_{i}.wav\tSentence {i}\t{dialect}\n"
        input_tsv.write_text(content)

        output_dir = temp_dir / "splits"
        output_dir.mkdir()

        create_splits(str(input_tsv), str(output_dir), data_root=str(temp_dir))

        # Load each split
        for split in ["train.tsv", "val.tsv", "test.tsv"]:
            df = load_swiss_german_metadata(str(output_dir / split))

            assert len(df) > 0
            assert 'sentence' in df.columns
            assert 'accent' in df.columns
            assert 'audio_path' in df.columns
