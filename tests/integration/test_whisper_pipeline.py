"""Integration test: real FHNW batch through the Whisper collator.

Requires the data volume (FHNW clips + metadata) and network/cache access for
the Whisper processor, so it is skipped when those are unavailable. Run via
``docker compose run --rm test-integration``.
"""

import pytest

import src.config as config


@pytest.mark.integration
def test_real_fhnw_batch_through_collator():
    pytest.importorskip("librosa")
    from transformers import WhisperConfig, WhisperProcessor

    train_tsv = config.DATA_DIR / "metadata" / "train.tsv"
    clips_dir = config.FHNW_SWISS_GERMAN_ROOT / "clips"
    if not train_tsv.exists() or not clips_dir.exists():
        pytest.skip("FHNW data volume not mounted")

    from src.data.whisper_collator import DataCollatorSpeechSeq2SeqWithPadding
    from src.data.whisper_dataset import WhisperSpeechDataset, load_metadata_df

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    processor.tokenizer.set_prefix_tokens(language="de", task="transcribe")
    cfg = WhisperConfig.from_pretrained("openai/whisper-large-v2")

    df = load_metadata_df(train_tsv).head(16)
    ds = WhisperSpeechDataset(df, clips_dir, processor, sampling_rate=16000,
                              dialect_column="accent")
    batch = [ds[i] for i in range(16)]

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=cfg.decoder_start_token_id
    )
    out = collator(batch)

    assert tuple(out["input_features"].shape) == (16, 80, 3000)
    assert out["labels"].shape[0] == 16
    assert (out["labels"] == -100).any().item()
    # Leading start-of-transcript token must have been stripped.
    assert (out["labels"][:, 0] != cfg.decoder_start_token_id).all().item()
