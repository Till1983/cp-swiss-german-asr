"""Unit tests for the Whisper seq2seq data collator.

These use lightweight stub feature-extractor / tokenizer objects so the
collator's own logic (feature stacking, -100 label masking, and the
start-of-transcript strip) is tested deterministically with no model/processor
download.
"""

import pytest
import torch

from src.data.whisper_collator import DataCollatorSpeechSeq2SeqWithPadding


class _StubFeatureExtractor:
    def pad(self, features, return_tensors="pt"):
        feats = torch.stack(
            [torch.as_tensor(f["input_features"], dtype=torch.float32) for f in features]
        )
        return {"input_features": feats}


class _LabelsBatch(dict):
    """dict that also exposes ``.attention_mask`` like a BatchEncoding."""

    attention_mask = None


class _StubTokenizer:
    pad_token_id = 0

    def pad(self, features, return_tensors="pt"):
        ids = [list(f["input_ids"]) for f in features]
        max_len = max(len(x) for x in ids)
        padded, mask = [], []
        for x in ids:
            pad_n = max_len - len(x)
            padded.append(x + [self.pad_token_id] * pad_n)
            mask.append([1] * len(x) + [0] * pad_n)
        out = _LabelsBatch(input_ids=torch.tensor(padded))
        out.attention_mask = torch.tensor(mask)
        return out


class _StubProcessor:
    def __init__(self):
        self.feature_extractor = _StubFeatureExtractor()
        self.tokenizer = _StubTokenizer()


@pytest.mark.unit
def test_input_features_are_stacked():
    processor = _StubProcessor()
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    features = [
        {"input_features": torch.zeros(80, 3000), "labels": [5, 6, 7]},
        {"input_features": torch.ones(80, 3000), "labels": [5, 8]},
    ]
    batch = collator(features)
    assert batch["input_features"].shape == (2, 80, 3000)


@pytest.mark.unit
def test_labels_padded_with_minus_100():
    processor = _StubProcessor()
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    features = [
        {"input_features": torch.zeros(80, 3000), "labels": [10, 11, 12]},
        {"input_features": torch.zeros(80, 3000), "labels": [10, 11]},
    ]
    batch = collator(features)
    labels = batch["labels"]
    assert labels.shape == (2, 3)
    # The short sequence's padded position must be masked to -100.
    assert labels[1, 2].item() == -100
    # Real tokens are preserved (no padding-token leakage).
    assert labels[0].tolist() == [10, 11, 12]


@pytest.mark.unit
def test_leading_start_token_is_stripped_when_present():
    processor = _StubProcessor()
    sot = 50258  # decoder_start_token_id
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=sot
    )
    features = [
        {"input_features": torch.zeros(80, 3000), "labels": [sot, 1, 2, 3]},
        {"input_features": torch.zeros(80, 3000), "labels": [sot, 4, 5, 6]},
    ]
    batch = collator(features)
    # Leading SOT dropped -> width 3, and first column is no longer the SOT.
    assert batch["labels"].shape == (2, 3)
    assert (batch["labels"][:, 0] != sot).all().item()


@pytest.mark.unit
def test_leading_token_not_stripped_when_not_uniform():
    processor = _StubProcessor()
    sot = 50258
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=sot
    )
    features = [
        {"input_features": torch.zeros(80, 3000), "labels": [sot, 1, 2]},
        {"input_features": torch.zeros(80, 3000), "labels": [99, 4, 5]},
    ]
    batch = collator(features)
    # Not all rows start with SOT -> nothing stripped (width stays 3).
    assert batch["labels"].shape == (2, 3)
    assert batch["labels"][0, 0].item() == sot
