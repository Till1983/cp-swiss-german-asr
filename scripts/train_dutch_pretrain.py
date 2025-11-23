import os
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import logging
from pathlib import Path
import torch
from datasets import Dataset
from evaluate import load
from src.data.loader import load_swiss_german_metadata, load_audio
from src.models.wav2vec2_model import Wav2Vec2Model
from src.data.collator import AudioDataCollatorCTC
from src.config import DUTCH_CV_ROOT, MODELS_DIR, RESULTS_DIR
import yaml

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
    EarlyStoppingCallback
)

# --- Compatibility shim for accelerate / transformers version mismatch ---
try:
    from accelerate import Accelerator
    import inspect

    # If unwrap_model doesn't accept keep_torch_compile, wrap it so Trainer calls succeed.
    sig = inspect.signature(Accelerator.unwrap_model)
    if "keep_torch_compile" not in sig.parameters:
        _orig_unwrap = Accelerator.unwrap_model

        def _compat_unwrap(self, model, *args, keep_torch_compile=False, **kwargs):
            # forward to original, ignore keep_torch_compile if not supported
            return _orig_unwrap(self, model, *args, **kwargs)

        Accelerator.unwrap_model = _compat_unwrap
        logger = logging.getLogger("train_dutch_pretrain")
        logger.info("Patched accelerate.Accelerator.unwrap_model to accept keep_torch_compile for compatibility.")
except Exception as _e:
    # If accelerate isn't available or introspection fails, proceed without shim.
    # Trainer will surface meaningful errors later if incompatible.
    pass

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
MODEL_NAME = "aware-ai/wav2vec2-large-xlsr-53-german-with-lm" # Use German
METADATA_FILE = DUTCH_CV_ROOT / "train.tsv"
AUDIO_DIR = DUTCH_CV_ROOT / "clips"
OUTPUT_DIR = MODELS_DIR / "pretrained" / "wav2vec2-dutch-cv"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_LOG = RESULTS_DIR / "metrics" / "dutch_pretrain.log"

# Load config
with open("configs/training/dutch_pretrain.yml", "r") as f:
    config = yaml.safe_load(f)

# Replace hardcoded TRAIN_ARGS with:
TRAIN_ARGS = config["training"]
TRAIN_ARGS["output_dir"] = str(OUTPUT_DIR)
TRAIN_ARGS["logging_dir"] = str(OUTPUT_DIR / "logs")

# --- Ensure compatibility with transformers.TrainingArguments ---
# Map legacy key
if "num_epochs" in TRAIN_ARGS and "num_train_epochs" not in TRAIN_ARGS:
    logger.info("Mapping 'num_epochs' -> 'num_train_epochs'")
    TRAIN_ARGS["num_train_epochs"] = TRAIN_ARGS.pop("num_epochs")

# Remove keys that belong to model config or are unsupported by TrainingArguments
for bad_key in [
    "dropout", "attention_dropout", "activation_dropout",
    "feat_proj_dropout", "layerdrop", "fp16_opt_level"
]:
    if bad_key in TRAIN_ARGS:
        logger.info(f"Removing '{bad_key}' from TRAIN_ARGS (not a TrainingArguments param).")
        TRAIN_ARGS.pop(bad_key)

# -----------------------------------------------------------------------------
# Device and fp16 setup
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
else:
    device = "cpu"
logger.info(f"Using device: {device}")

# Override fp16 if not supported
if (device != "cuda") and TRAIN_ARGS.get("fp16", False):
    logger.warning(f"{device.upper()} does not support fp16 (mixed precision). Disabling fp16 for training.")
    TRAIN_ARGS["fp16"] = False

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

    # Load model and processor ONCE
    try:
        model_wrapper = Wav2Vec2Model(model_name=MODEL_NAME)
        model = model_wrapper.model.to(device)
        processor = model_wrapper.processor
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        sys.exit(1)

    # --- Vocab check (use dataset text) and preprocess dataset to provide
    #     the columns Trainer expects: 'input_values' and 'labels' ---
    # Simple tokenizer sanity check: ensure tokenizer produces tokens for example sentences
    try:
        sample_count = min(5, len(train_dataset))
        for row in train_dataset.select(range(sample_count)):
            sent = row.get("sentence", "")
            tokenized = processor.tokenizer(sent, add_special_tokens=False)
            if not tokenized.input_ids:
                logger.warning("Tokenizer produced no tokens for a sample sentence (possible mismatch).")
    except Exception as e:
        logger.warning(f"Tokenizer sanity check skipped: {e}")

    # Create input_values and labels columns expected by Trainer.
    sampling_rate = config.get("data", {}).get("sampling_rate", 16000)

    def prepare_examples(batch):
        # normalize audio entries to raw arrays
        audio_list = []
        for a in batch["audio"]:
            if isinstance(a, dict) and "array" in a:
                audio_list.append(a["array"])
            else:
                audio_list.append(a)

        # processor(audio_list, ...) returns lists of input_values when batched
        inputs = processor(audio_list, sampling_rate=sampling_rate, padding=True, return_attention_mask=False)
        batch["input_values"] = inputs["input_values"]

        # tokenise transcripts -> label ids (no special tokens for CTC)
        tokenized = processor.tokenizer(batch["sentence"], add_special_tokens=False)
        # ensure labels is a list of lists
        batch["labels"] = tokenized["input_ids"]
        return batch

    # Map in batched mode to add columns; keep existing columns for debug visibility
    train_dataset = train_dataset.map(prepare_examples, batched=True, batch_size=4)
    logger.info("Dataset columns after preprocessing: %s", train_dataset.column_names)

    # Ensure training mode
    model.train()

    # Data collator for CTC
    data_collator = AudioDataCollatorCTC(processor=processor, padding=True)

    # Metric for evaluation
    wer_metric = load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions.argmax(-1)
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Training arguments
    logger.info("TRAIN_ARGS keys before TrainingArguments: %s", list(TRAIN_ARGS.keys()))
    logger.debug("TRAIN_ARGS content: %s", TRAIN_ARGS)

    # --- Sanitize TRAIN_ARGS types to avoid type-comparison errors in TrainingArguments ---
    # Define expected numeric/bool keys and coerce strings to proper types if needed.
    int_keys = {
        "num_train_epochs",
        "warmup_steps",
        "max_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
    }
    float_keys = {
        "learning_rate",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "max_grad_norm",
    }
    bool_keys = {
        "fp16",
    }

    for k in int_keys:
        if k in TRAIN_ARGS:
            v = TRAIN_ARGS[k]
            if isinstance(v, str):
                try:
                    TRAIN_ARGS[k] = int(v)
                    logger.info(f"Coerced TRAIN_ARGS['{k}'] from str -> int ({TRAIN_ARGS[k]})")
                except Exception:
                    logger.warning(f"Failed to coerce TRAIN_ARGS['{k}']='{v}' to int; leaving as-is")

    for k in float_keys:
        if k in TRAIN_ARGS:
            v = TRAIN_ARGS[k]
            if isinstance(v, str):
                try:
                    TRAIN_ARGS[k] = float(v)
                    logger.info(f"Coerced TRAIN_ARGS['{k}'] from str -> float ({TRAIN_ARGS[k]})")
                except Exception:
                    logger.warning(f"Failed to coerce TRAIN_ARGS['{k}']='{v}' to float; leaving as-is")

    for k in bool_keys:
        if k in TRAIN_ARGS:
            v = TRAIN_ARGS[k]
            if isinstance(v, str):
                TRAIN_ARGS[k] = v.strip().lower() in ("1", "true", "yes", "y")
                logger.info(f"Coerced TRAIN_ARGS['{k}'] from str -> bool ({TRAIN_ARGS[k]})")

    # Small sanity checks
    if "max_steps" in TRAIN_ARGS:
        try:
            TRAIN_ARGS["max_steps"] = int(TRAIN_ARGS["max_steps"])
        except Exception:
            pass

    logger.debug("TRAIN_ARGS after sanitization: %s", TRAIN_ARGS)

    training_args = TrainingArguments(**TRAIN_ARGS)

    # Trainer setup
    eval_dataset = None  # No validation set for pre-training

    # Only enable early stopping if we have a validation set (and metric_for_best_model is set).
    callbacks = []
    if eval_dataset is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
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