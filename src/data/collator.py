import torch
from transformers import Wav2Vec2Processor, WhisperProcessor

class AudioDataCollatorCTC:
    """
    Data collator for dynamic padding of audio sequences for CTC-based ASR models.
    Handles batch processing, attention masks, and labels for CTC loss.
    Compatible with Whisper, Wav2Vec2, and MMS models.
    """

    def __init__(self, processor, padding=True, max_length=None, pad_to_multiple_of=None):
        """
        Args:
            processor: HuggingFace processor (Wav2Vec2Processor, WhisperProcessor, etc.)
            padding: Whether to pad sequences dynamically
            max_length: Optional max length for padding
            pad_to_multiple_of: Optional pad length to a multiple of this value
        """
        self.processor = processor
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        """
        Collate a batch of features for ASR training/evaluation.

        Args:
            features: List of dicts with keys 'input_values' (audio), 'labels' (transcription)

        Returns:
            Dict of batched tensors: input_values, attention_mask, labels
        """
        # Extract input_values and labels
        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]

        # Pad input_values (audio)
        batch = self.processor.pad(
            {"input_values": input_values},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        # Pad labels (transcriptions)
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                {"input_ids": labels},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt"
            )

        # Replace padding with -100 for CTC loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def get_processor_for_model(model_type, model_name):
    """
    Utility to get the correct processor for a given model type/name.
    """
    if model_type == "wav2vec2":
        return Wav2Vec2Processor.from_pretrained(model_name)
    elif model_type == "whisper":
        return WhisperProcessor.from_pretrained(f"openai/whisper-{model_name}")
    elif model_type == "mms":
        # MMS uses Wav2Vec2Processor for ASR
        return Wav2Vec2Processor.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")