"""Whisper-specific data collator and example preparation.

This is the sequence-to-sequence padding scheme for Whisper, and is
deliberately NOT a reuse of ``src/data/collator.py``'s ``AudioDataCollatorCTC``
(that one is CTC-shaped: it pads raw ``input_values`` and masks labels for the
CTC loss, which is the wrong padding/label scheme for an encoder-decoder model
like Whisper).

Responsibilities (per the Whisper fine-tuning recipe):

* ``input_features`` are produced by the feature extractor and are always
  80 x 3000 (Whisper pads/truncates every clip to 30 s of log-mel frames
  regardless of the clip's true length), so the feature side only needs to be
  stacked into a batch.
* ``labels`` are produced by the tokenizer and padded with ``-100`` so the
  padded positions are ignored by the cross-entropy loss.
* The Whisper tokenizer prepends the start-of-transcript token
  (``decoder_start_token_id``) to every label sequence. The model re-adds that
  token when it builds ``decoder_input_ids`` via ``shift_tokens_right``, so the
  collator strips the single leading start token to avoid an off-by-one label
  misalignment (a doubled SOT). The language/task prefix tokens are kept --
  they are part of what the model learns to predict.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


def prepare_whisper_example(
    processor,
    audio,
    sentence: str,
    sampling_rate: int = 16000,
) -> Dict[str, Any]:
    """Turn one (audio, transcript) pair into Whisper model inputs.

    Returns a dict with ``input_features`` (80 x 3000 log-mel features) and
    ``labels`` (token ids, including the tokenizer's prefix tokens). The
    collator below pads/stacks a list of these into a batch.
    """
    features = processor.feature_extractor(
        audio, sampling_rate=sampling_rate
    ).input_features[0]
    labels = processor.tokenizer(sentence).input_ids
    return {"input_features": features, "labels": labels}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Collate pre-extracted Whisper features + tokenized labels into a batch.

    Args:
        processor: A ``WhisperProcessor`` (needs ``feature_extractor`` and
            ``tokenizer`` with ``.pad``).
        decoder_start_token_id: The model's ``config.decoder_start_token_id``
            (the start-of-transcript token). When all label sequences start
            with this token, the leading token is stripped. If ``None``, no
            stripping is performed.
    """

    processor: Any
    decoder_start_token_id: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # --- audio side: input_features are all 80 x 3000, just stack them ---
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # --- label side: pad to longest, then mask pads with -100 ---
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If every sequence begins with the start-of-transcript token, drop it:
        # the model prepends it again when forming decoder_input_ids.
        if self.decoder_start_token_id is not None and labels.shape[1] > 0:
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
