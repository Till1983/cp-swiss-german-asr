"""Unit tests for audio utility functions."""
import pytest
from pathlib import Path
from unittest.mock import patch
from src.utils.audio_utils import validate_audio_file, get_audio_duration


class TestValidateAudioFile:
    """Test suite for validate_audio_file function."""

    @pytest.mark.unit
    def test_valid_wav_file(self, sample_audio_path):
        """Test validation passes for valid .wav file."""
        assert validate_audio_file(str(sample_audio_path)) is True

    @pytest.mark.unit
    def test_nonexistent_file(self):
        """Test validation fails for non-existent file."""
        assert validate_audio_file("nonexistent.wav") is False

    @pytest.mark.unit
    @pytest.mark.parametrize("extension", [".flac", ".wav", ".mp3"])
    def test_valid_extensions(self, temp_dir, extension):
        """Test all valid audio extensions are accepted."""
        audio_file = temp_dir / f"test{extension}"
        audio_file.touch()

        assert validate_audio_file(str(audio_file)) is True

    @pytest.mark.unit
    @pytest.mark.parametrize("extension", [".txt", ".mp4", ".pdf", ".json", ".py"])
    def test_invalid_extensions(self, temp_dir, extension):
        """Test invalid extensions are rejected."""
        file = temp_dir / f"test{extension}"
        file.touch()

        assert validate_audio_file(str(file)) is False

    @pytest.mark.unit
    def test_file_without_extension(self, temp_dir):
        """Test file without extension is rejected."""
        file = temp_dir / "noextension"
        file.touch()

        assert validate_audio_file(str(file)) is False

    @pytest.mark.unit
    def test_handles_exception_gracefully(self):
        """Test function handles exceptions and returns False."""
        # Invalid path that causes exception
        assert validate_audio_file("\0invalid") is False

    @pytest.mark.unit
    def test_case_insensitive_extension(self, temp_dir):
        """Test extension check is case-insensitive."""
        wav_upper = temp_dir / "test.WAV"
        wav_upper.touch()

        assert validate_audio_file(str(wav_upper)) is True

    @pytest.mark.unit
    def test_empty_path(self):
        """Test empty path returns False."""
        assert validate_audio_file("") is False

    @pytest.mark.unit
    def test_path_with_spaces(self, temp_dir):
        """Test path with spaces is handled correctly."""
        audio_file = temp_dir / "test file with spaces.wav"
        audio_file.touch()

        assert validate_audio_file(str(audio_file)) is True

    @pytest.mark.unit
    def test_path_object_input(self, sample_audio_path):
        """Test Path object can be converted to string and validated."""
        # The function expects str, so we pass str(Path)
        assert validate_audio_file(str(sample_audio_path)) is True


class TestGetAudioDuration:
    """Test suite for get_audio_duration function."""

    @pytest.mark.unit
    def test_returns_duration_for_valid_file(self, sample_audio_path):
        """Test returns duration for valid audio file."""
        duration = get_audio_duration(str(sample_audio_path))

        assert duration is not None
        assert isinstance(duration, float)
        assert duration > 0

    @pytest.mark.unit
    def test_returns_correct_duration(self, sample_audio_path):
        """Test returned duration is approximately correct."""
        duration = get_audio_duration(str(sample_audio_path))

        # Our sample audio is 1 second
        assert 0.9 <= duration <= 1.1

    @pytest.mark.unit
    def test_returns_none_for_nonexistent_file(self):
        """Test returns None for non-existent file."""
        duration = get_audio_duration("nonexistent.wav")
        assert duration is None

    @pytest.mark.unit
    @patch('librosa.get_duration', side_effect=Exception("Error"))
    def test_handles_librosa_error(self, mock_duration):
        """Test handles librosa errors gracefully."""
        duration = get_audio_duration("dummy.wav")
        assert duration is None

    @pytest.mark.unit
    def test_returns_none_for_invalid_file(self, temp_dir):
        """Test returns None for invalid audio file (empty or corrupt)."""
        invalid_file = temp_dir / "invalid.wav"
        invalid_file.write_text("not audio data")

        duration = get_audio_duration(str(invalid_file))
        assert duration is None

    @pytest.mark.unit
    def test_duration_type_is_float(self, sample_audio_path):
        """Test duration is returned as float type."""
        duration = get_audio_duration(str(sample_audio_path))

        assert isinstance(duration, float)

    @pytest.mark.unit
    @patch('librosa.get_duration', return_value=5.5)
    def test_returns_librosa_duration(self, mock_duration):
        """Test returns the duration from librosa."""
        duration = get_audio_duration("test.wav")

        assert duration == 5.5

    @pytest.mark.unit
    def test_empty_path(self):
        """Test empty path returns None."""
        duration = get_audio_duration("")
        assert duration is None
