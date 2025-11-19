import os
import sys
import logging
from pathlib import Path
import torch
from datasets import Dataset, load_metric
from src.data.loader import load_swiss_german_metadata, load_audio
from src.models.wav2vec2_model import Wav2Vec2Model
from src.config import DUTCH_CV_ROOT, MODELS_DIR, RESULTS_DIR

"""
Dutch ASR Pre-training Script

This script loads the Dutch Common Voice corpus, prepares it for ASR pre-training,
initializes a Wav2Vec2 model, and trains it using Hugging Face Trainer.
It includes progress logging, error handling, and saves checkpoints after each epoch.

Usage:
    python scripts/train_dutch_pretrain.py

Environment/configuration is managed via src/config.py.
"""


from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorCTCTokenizer,
)

# Import project modules

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("train_dutch_pretrain")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53-german" # Use German
METADATA_FILE = DUTCH_CV_ROOT / "validated.tsv"
AUDIO_DIR = DUTCH_CV_ROOT / "clips"
OUTPUT_DIR = MODELS_DIR / "pretrained" / "wav2vec2-dutch-cv"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_LOG = RESULTS_DIR / "metrics" / "dutch_pretrain.log"

# Training hyperparameters
TRAIN_ARGS = {
    "output_dir": str(OUTPUT_DIR),
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "num_train_epochs": 5,
    "learning_rate": 3e-4,
    "logging_dir": str(OUTPUT_DIR / "logs"),
    "logging_steps": 10,
    "save_total_limit": 3,
    "fp16": torch.cuda.is_available(),
    "load_best_model_at_end": True,
    "metric_for_best_model": "loss",
    "greater_is_better": False,
}

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def prepare_dataset(metadata_path, audio_dir, limit=None):
    """
    Prepare Dutch Common Voice dataset for ASR training.

    Args:
        metadata_path: Path to TSV metadata file.
        audio_dir: Directory containing audio files.
        limit: Optional limit on number of samples.

    Returns:
        Hugging Face Dataset object.
    """
    try:
        df = load_swiss_german_metadata(str(metadata_path))
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        raise

    # Dutch CV uses 'sentence' and 'path' columns
    required_columns = {'sentence', 'path'}
    if not required_columns.issubset(df.columns):
        logger.error(f"Metadata missing required columns: {required_columns}")
        raise ValueError("Invalid metadata file format.")

    if limit:
        df = df.head(limit)

    # Add absolute audio paths
    df['audio_path'] = df['path'].apply(lambda x: str(audio_dir / x))

    # Remove samples with missing audio files
    df = df[df['audio_path'].apply(lambda p: Path(p).exists())]

    # Prepare Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Add audio loading
    def map_audio(batch):
        try:
            batch["audio"] = load_audio(batch["audio_path"])
        except Exception as e:
            logger.warning(f"Audio load failed for {batch['audio_path']}: {e}")
            batch["audio"] = None
        return batch

    dataset = dataset.map(map_audio)
    dataset = dataset.filter(lambda x: x["audio"] is not None)

    logger.info(f"Prepared dataset with {len(dataset)} samples.")
    return dataset

# -----------------------------------------------------------------------------
# Main Training Routine
# -----------------------------------------------------------------------------
def main():
    logger.info("Dutch ASR Pre-training Script Started.")

    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    try:
        train_dataset = prepare_dataset(METADATA_FILE, AUDIO_DIR)
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        sys.exit(1)

    # Load model and processor
    try:
        model_wrapper = Wav2Vec2Model(model_name=MODEL_NAME)
        model = model_wrapper.model
        processor = model_wrapper.processor
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        sys.exit(1)

    # Data collator for CTC
    data_collator = DataCollatorCTCTokenizer(processor=processor, padding=True)

    # Metric for evaluation
    wer_metric = load_metric("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions.argmax(-1)
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Training arguments
    training_args = TrainingArguments(**TRAIN_ARGS)

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # No validation set for pre-training
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Training loop with checkpointing
    try:
        logger.info("Starting training...")
        train_result = trainer.train()
        logger.info("Training completed.")
        trainer.save_model(str(OUTPUT_DIR))
        trainer.save_state()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    # Save training logs
    try:
        with open(RESULTS_LOG, "w") as f:
            f.write(str(train_result.metrics))
        logger.info(f"Training metrics saved to {RESULTS_LOG}")
    except Exception as e:
        logger.warning(f"Failed to save training metrics: {e}")

    logger.info("Dutch ASR Pre-training Script Finished.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)