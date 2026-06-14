"""Unit tests for Whisper setup helpers (path resolution + use_cache/Requirement D)."""

from pathlib import Path

import pytest
import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration

from src.training.whisper_setup import (
    configure_caching,
    resolve_path,
    set_decoding_config,
)


def tiny_model():
    cfg = WhisperConfig(
        d_model=32, encoder_layers=2, decoder_layers=2,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=64, decoder_ffn_dim=64,
        num_mel_bins=80, max_source_positions=50, max_target_positions=50,
        vocab_size=120, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        decoder_start_token_id=1,
    )
    return WhisperForConditionalGeneration(cfg)


@pytest.mark.unit
def test_resolve_path_joins_relative():
    assert resolve_path("/base/data", "metadata/train.tsv") == Path("/base/data/metadata/train.tsv")


@pytest.mark.unit
def test_resolve_path_passthrough_absolute():
    assert resolve_path("/base/data", "/abs/override.pt") == Path("/abs/override.pt")


@pytest.mark.unit
def test_configure_caching_gc_on_disables_train_cache():
    model = tiny_model()
    configure_caching(model, gradient_checkpointing=True)
    assert model.config.use_cache is False
    # Generation cache stays enabled regardless of the training-time setting.
    assert model.generation_config.use_cache is True


@pytest.mark.unit
def test_configure_caching_gc_off_keeps_train_cache():
    model = tiny_model()
    configure_caching(model, gradient_checkpointing=False)
    assert model.config.use_cache is True
    assert model.generation_config.use_cache is True


@pytest.mark.unit
def test_generate_works_when_train_cache_disabled():
    """Requirement D: generate() re-enables the cache regardless of GC setting."""
    model = tiny_model()
    configure_caching(model, gradient_checkpointing=True)
    assert model.config.use_cache is False

    input_features = torch.randn(1, 80, 100)  # 2 * max_source_positions
    out = model.generate(input_features=input_features, max_new_tokens=3)
    assert out.shape[0] == 1
    assert out.shape[1] >= 1
    # generate did not flip the training-time config back on by side effect
    assert model.config.use_cache is False
    assert model.generation_config.use_cache is True


class _StubTokenizer:
    def __init__(self):
        self.prefix = None

    def set_prefix_tokens(self, language=None, task=None):
        self.prefix = (language, task)


class _StubProcessor:
    def __init__(self):
        self.tokenizer = _StubTokenizer()

    def get_decoder_prompt_ids(self, language=None, task=None):
        return [(1, 50259), (2, 50359), (3, 50363)]


@pytest.mark.unit
def test_set_decoding_config_forces_language_and_disables_prev_tokens():
    model = tiny_model()
    processor = _StubProcessor()
    set_decoding_config(model, processor, language="de", task="transcribe",
                        condition_on_prev_tokens=False)
    assert processor.tokenizer.prefix == ("de", "transcribe")
    assert model.config.forced_decoder_ids == [(1, 50259), (2, 50359), (3, 50363)]
    assert model.config.condition_on_prev_tokens is False
    assert model.generation_config.condition_on_prev_tokens is False
