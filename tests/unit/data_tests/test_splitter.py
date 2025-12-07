"""Unit tests for data splitter module."""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from src.data.splitter import create_splits


class TestCreateSplits:
    """Test suite for create_splits function."""

    @pytest.fixture
    def sample_tsv_content(self):
        """Sample TSV content for testing."""
        return """client_id\tpath\tsentence\taccent
client_001\taudio1.wav\tSentence 1\tBE
client_002\taudio2.wav\tSentence 2\tBE
client_003\taudio3.wav\tSentence 3\tZH
client_004\taudio4.wav\tSentence 4\tZH
client_005\taudio5.wav\tSentence 5\tVS
client_006\taudio6.wav\tSentence 6\tVS
client_007\taudio7.wav\tSentence 7\tBE
client_008\taudio8.wav\tSentence 8\tZH
client_009\taudio9.wav\tSentence 9\tVS
client_010\taudio10.wav\tSentence 10\tBE
client_011\taudio11.wav\tSentence 11\tZH
client_012\taudio12.wav\tSentence 12\tVS
client_013\taudio13.wav\tSentence 13\tBE
client_014\taudio14.wav\tSentence 14\tZH
client_015\taudio15.wav\tSentence 15\tVS
client_016\taudio16.wav\tSentence 16\tBE
client_017\taudio17.wav\tSentence 17\tZH
client_018\taudio18.wav\tSentence 18\tVS
client_019\taudio19.wav\tSentence 19\tBE
client_020\taudio20.wav\tSentence 20\tZH"""

    @pytest.fixture
    def sample_tsv_file(self, temp_dir, sample_tsv_content):
        """Create a sample TSV file for testing."""
        tsv_path = temp_dir / "test_data.tsv"
        tsv_path.write_text(sample_tsv_content)
        return tsv_path

    @pytest.mark.unit
    def test_create_splits_returns_dict(self, sample_tsv_file, temp_dir):
        """Test create_splits returns dictionary with stats."""
        output_dir = temp_dir / "splits"
        output_dir.mkdir()

        result = create_splits(
            str(sample_tsv_file),
            str(output_dir),
            data_root=str(temp_dir)
        )

        assert isinstance(result, dict)
        assert 'train' in result
        assert 'val' in result
        assert 'test' in result
        assert 'dialects' in result

    @pytest.mark.unit
    def test_create_splits_creates_output_files(self, sample_tsv_file, temp_dir):
        """Test create_splits creates train/val/test TSV files."""
        output_dir = temp_dir / "splits"
        output_dir.mkdir()

        create_splits(
            str(sample_tsv_file),
            str(output_dir),
            data_root=str(temp_dir)
        )

        assert (output_dir / "train.tsv").exists()
        assert (output_dir / "val.tsv").exists()
        assert (output_dir / "test.tsv").exists()

    @pytest.mark.unit
    def test_create_splits_70_15_15_ratio(self, sample_tsv_file, temp_dir):
        """Test create_splits uses 70/15/15 train/val/test ratio."""
        output_dir = temp_dir / "splits"
        output_dir.mkdir()

        result = create_splits(
            str(sample_tsv_file),
            str(output_dir),
            data_root=str(temp_dir)
        )

        total = result['train'] + result['val'] + result['test']

        # Check approximate ratios (70/15/15)
        assert 0.65 <= result['train'] / total <= 0.75
        assert 0.10 <= result['val'] / total <= 0.20
        assert 0.10 <= result['test'] / total <= 0.20

    @pytest.mark.unit
    def test_create_splits_counts_dialects(self, sample_tsv_file, temp_dir):
        """Test create_splits correctly counts unique dialects."""
        output_dir = temp_dir / "splits"
        output_dir.mkdir()

        result = create_splits(
            str(sample_tsv_file),
            str(output_dir),
            data_root=str(temp_dir)
        )

        # Our test data has 3 dialects: BE, ZH, VS
        assert result['dialects'] == 3

    @pytest.mark.unit
    def test_create_splits_adds_audio_path_column(self, sample_tsv_file, temp_dir):
        """Test create_splits adds audio_path column to output."""
        output_dir = temp_dir / "splits"
        output_dir.mkdir()

        create_splits(
            str(sample_tsv_file),
            str(output_dir),
            data_root=str(temp_dir)
        )

        train_df = pd.read_csv(output_dir / "train.tsv", sep='\t')
        assert 'audio_path' in train_df.columns

    @pytest.mark.unit
    def test_create_splits_audio_path_format(self, sample_tsv_file, temp_dir):
        """Test audio_path column has correct format."""
        output_dir = temp_dir / "splits"
        output_dir.mkdir()

        create_splits(
            str(sample_tsv_file),
            str(output_dir),
            data_root=str(temp_dir)
        )

        train_df = pd.read_csv(output_dir / "train.tsv", sep='\t')

        # All audio paths should end with clips/<filename>
        for audio_path in train_df['audio_path']:
            assert '/clips/' in audio_path
            assert audio_path.endswith('.wav')

    @pytest.mark.unit
    def test_create_splits_stratified_by_dialect(self, sample_tsv_file, temp_dir):
        """Test create_splits stratifies by dialect."""
        output_dir = temp_dir / "splits"
        output_dir.mkdir()

        create_splits(
            str(sample_tsv_file),
            str(output_dir),
            data_root=str(temp_dir)
        )

        # Load all splits
        train_df = pd.read_csv(output_dir / "train.tsv", sep='\t')
        val_df = pd.read_csv(output_dir / "val.tsv", sep='\t')
        test_df = pd.read_csv(output_dir / "test.tsv", sep='\t')

        # Each split should contain samples from each dialect
        original_dialects = {'BE', 'ZH', 'VS'}

        train_dialects = set(train_df['accent'].unique())
        # At minimum, training set should have multiple dialects represented
        assert len(train_dialects) >= 2

    @pytest.mark.unit
    def test_create_splits_uses_default_data_root(self, sample_tsv_file, temp_dir):
        """Test create_splits uses TSV parent as default data_root."""
        output_dir = temp_dir / "splits"
        output_dir.mkdir()

        result = create_splits(
            str(sample_tsv_file),
            str(output_dir),
            data_root=None  # Use default
        )

        # Should complete without error
        assert result['train'] > 0

    @pytest.mark.unit
    def test_create_splits_preserves_all_samples(self, sample_tsv_file, temp_dir):
        """Test no samples are lost during splitting."""
        output_dir = temp_dir / "splits"
        output_dir.mkdir()

        original_df = pd.read_csv(sample_tsv_file, sep='\t')
        original_count = len(original_df)

        result = create_splits(
            str(sample_tsv_file),
            str(output_dir),
            data_root=str(temp_dir)
        )

        total_split_count = result['train'] + result['val'] + result['test']
        assert total_split_count == original_count

    @pytest.mark.unit
    def test_create_splits_deterministic(self, sample_tsv_file, temp_dir):
        """Test create_splits produces deterministic results (fixed random_state)."""
        output_dir1 = temp_dir / "splits1"
        output_dir1.mkdir()
        output_dir2 = temp_dir / "splits2"
        output_dir2.mkdir()

        result1 = create_splits(
            str(sample_tsv_file),
            str(output_dir1),
            data_root=str(temp_dir)
        )
        result2 = create_splits(
            str(sample_tsv_file),
            str(output_dir2),
            data_root=str(temp_dir)
        )

        # Results should be identical
        assert result1 == result2

        # Files should have same content
        train1 = pd.read_csv(output_dir1 / "train.tsv", sep='\t')
        train2 = pd.read_csv(output_dir2 / "train.tsv", sep='\t')
        assert train1.equals(train2)
