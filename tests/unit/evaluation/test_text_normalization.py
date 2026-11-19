"""Tests for text normalization input validation."""

import pytest

from src.evaluation.text_normalization import normalize_text


@pytest.mark.unit
def test_normalization_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Invalid mode 'unknown'"):
        normalize_text('Hello!', mode='unknown')
