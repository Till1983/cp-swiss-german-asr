import os
import sys
import logging

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from datasets import Dataset
import evaluate  # FIXED: Use evaluate instead of deprecated load_metric
from src.data.loader import load_swiss_german_metadata, load_audio
from src.models.wav2vec2_model import Wav2Vec2Model
from src.config import DUTCH_CV_ROOT, GERMAN_CV_ROOT, MODELS_DIR, RESULTS_DIR  # FIXED: Added DUTCH_CV_ROOT
import numpy as np
from torch.utils.data import DataLoader
import yaml
from transformers import Wav2Vec2ForCTC
from src.data.collator import AudioDataCollatorCTC

"""
German ASR Adaptation Script

This script loads the Dutch-pretrained Wav2Vec2 checkpoint, applies elastic weight consolidation (EWC)
to mitigate catastrophic forgetting, and fine-tunes the model on German Common Voice data.
It includes comprehensive logging, error handling, and saves the adapted model weights.

Usage:
    python scripts/train_german_adaptation.py --config configs/training/german_adaptation.yml

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
    _sig = inspect.signature(Accelerator.unwrap_model)
    if "keep_torch_compile" not in _sig.parameters:
        _orig_unwrap = Accelerator.unwrap_model

        def _compat_unwrap(self, model, *args, keep_torch_compile=False, **kwargs):
            return _orig_unwrap(self, model, *args, **kwargs)

        Accelerator.unwrap_model = _compat_unwrap
        logging.getLogger("train_german_adaptation").info(
            "Patched accelerate.Accelerator.unwrap_model to accept keep_torch_compile for compatibility."
        )
except Exception:
    pass

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("train_german_adaptation")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PRETRAINED_CHECKPOINT = MODELS_DIR / "pretrained" / "wav2vec2-dutch-pretrained"
MODEL_NAME = str(PRETRAINED_CHECKPOINT)
METADATA_FILE = GERMAN_CV_ROOT / "train.tsv"
AUDIO_DIR = GERMAN_CV_ROOT / "clips"
OUTPUT_DIR = MODELS_DIR / "adapted" / "wav2vec2-german-adapted"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_LOG = RESULTS_DIR / "metrics" / "german_adaptation.log"

# Load config from YAML
with open("configs/training/german_adaptation.yml", "r") as f:
    config = yaml.safe_load(f)

# Use checkpoint path from config
# If path is relative, prepend MODELS_DIR; if absolute, use as-is
checkpoint_from_config = config["model"]["dutch_checkpoint"]
if Path(checkpoint_from_config).is_absolute():
    PRETRAINED_CHECKPOINT = Path(checkpoint_from_config)
    logger.info(f"Using absolute checkpoint path: {PRETRAINED_CHECKPOINT}")
else:
    PRETRAINED_CHECKPOINT = MODELS_DIR / checkpoint_from_config
    logger.info(f"Using relative checkpoint path: {checkpoint_from_config} -> {PRETRAINED_CHECKPOINT}")
MODEL_NAME = str(PRETRAINED_CHECKPOINT)

TRAIN_ARGS = config["training"]
TRAIN_ARGS["output_dir"] = str(OUTPUT_DIR)
TRAIN_ARGS["logging_dir"] = str(OUTPUT_DIR / "logs")

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
# Elastic Weight Consolidation (EWC) Implementation
# -----------------------------------------------------------------------------
class EWCTrainer(Trainer):
    """
    Hugging Face Trainer subclass implementing Elastic Weight Consolidation (EWC).
    Penalizes changes to important parameters to prevent catastrophic forgetting.
    """
    def __init__(self, *args, ewc_lambda=0.4, fisher_dict=None, old_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.fisher_dict = fisher_dict
        self.old_params = old_params

    def compute_ewc_loss(self):
        """
        Compute EWC penalty based on Fisher information and parameter changes.
        """
        if self.fisher_dict is None or self.old_params is None:
            return 0.0
        ewc_loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict and name in self.old_params:
                fisher = self.fisher_dict[name]
                old_param = self.old_params[name]
                ewc_loss += (fisher * (param - old_param).pow(2)).sum()
        return self.ewc_lambda * ewc_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to add EWC penalty.
        """
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        ewc_loss = self.compute_ewc_loss()
        total_loss = loss + ewc_loss
        if return_outputs:
            return total_loss, outputs
        return total_loss

def compute_fisher_information(model, dataloader, device):
    """
    Estimate Fisher information for model parameters using provided dataloader.
    
    FIXED: Removed confusing num_samples iteration limit. Now processes entire dataloader.
    
    Args:
        model: The model to compute Fisher information for
        dataloader: DataLoader containing reference samples (e.g., Dutch data)
        device: Device to run computation on
    
    Returns:
        fisher_dict: Dictionary mapping parameter names to Fisher information
    """
    logger.info(f"Estimating Fisher information using {len(dataloader.dataset)} samples...")
    fisher_dict = {}
    model.eval()
    
    # Initialize Fisher dict with zeros
    for name, param in model.named_parameters():
        fisher_dict[name] = torch.zeros_like(param)
    
    # Compute gradients for each batch
    num_batches = len(dataloader)
    for i, batch in enumerate(dataloader):
        if i % 50 == 0:  # Progress logging
            logger.info(f"Fisher estimation: batch {i}/{num_batches}")
        
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items() 
            if k != "audio_path"
        }
        model.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        loss.backward()
        
        # Accumulate squared gradients (Fisher information)
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_dict[name] += param.grad.data.pow(2)
    
    # Average Fisher information across all batches
    for name in fisher_dict:
        fisher_dict[name] /= num_batches
    
    logger.info("Fisher information estimation complete.")
    return fisher_dict

def get_model_params(model):
    """
    Get a copy of model parameters for EWC reference.
    """
    return {name: param.clone().detach() for name, param in model.named_parameters()}

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def prepare_dataset(metadata_path, audio_dir, limit=None, random_sample=False):
    """
    Prepare German Common Voice dataset for ASR training.

    Args:
        metadata_path: Path to TSV metadata file.
        audio_dir: Directory containing audio files.
        limit: Optional limit on number of samples.
        random_sample: If True and limit is set, sample randomly instead of taking first N.

    Returns:
        Hugging Face Dataset object.
    """
    try:
        df = load_swiss_german_metadata(str(metadata_path))
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        raise

    required_columns = {'sentence', 'path'}
    if not required_columns.issubset(df.columns):
        logger.error(f"Metadata missing required columns: {required_columns}")
        raise ValueError("Invalid metadata file format.")

    if limit:
        if random_sample:
            df = df.sample(n=limit, random_state=42)
            logger.info(f"Randomly sampled {limit} examples from {len(df)} total")
        else:
            df = df.head(limit)
            logger.info(f"Using first {limit} examples from {len(df)} total")

    # Add absolute audio paths
    df['audio_path'] = df['path'].apply(lambda x: str(audio_dir / x))

    # Remove samples with missing audio files
    initial_count = len(df)
    df = df[df['audio_path'].apply(lambda p: Path(p).exists())]
    if len(df) < initial_count:
        logger.warning(f"Removed {initial_count - len(df)} samples with missing audio files")

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
    logger.info("German ASR Adaptation Script Started.")

    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Load German training dataset
    try:
        train_dataset = prepare_dataset(METADATA_FILE, AUDIO_DIR)
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        sys.exit(1)

    # Load pretrained Dutch model and processor
    try:
        model_wrapper = Wav2Vec2Model(model_name=MODEL_NAME)
        model = model_wrapper.model.to(device)
        processor = model_wrapper.processor
        logger.info(f"Loaded Dutch pretrained model from {MODEL_NAME}")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        sys.exit(1)

    # Tokenizer sanity check
    try:
        sample_count = 5
        for row in train_dataset.select(range(min(sample_count, len(train_dataset)))):
            sent = row.get("sentence", "")
            toks = processor.tokenizer(sent, add_special_tokens=False)
            if not getattr(toks, "input_ids", toks.get("input_ids", None)):
                logger.warning("Tokenizer produced no tokens for a sample sentence (possible tokenizer mismatch).")
    except Exception as e:
        logger.warning(f"Tokenizer sanity check skipped: {e}")

    # Preprocess train_dataset -> add 'input_values' and 'labels' expected by Trainer
    sampling_rate = config.get("data", {}).get("sampling_rate", 16000)

    def prepare_examples(batch):
        audio_list = []
        for a in batch["audio"]:
            if isinstance(a, dict) and "array" in a:
                audio_list.append(a["array"])
            else:
                audio_list.append(a)

        inputs = processor(audio_list, sampling_rate=sampling_rate, padding=True, return_attention_mask=False)
        batch["input_values"] = inputs["input_values"]

        tokenized = processor.tokenizer(batch["sentence"], add_special_tokens=False)
        batch["labels"] = tokenized["input_ids"]
        return batch

    train_dataset = train_dataset.map(prepare_examples, batched=True, batch_size=4)
    logger.info("Train dataset columns after preprocessing: %s", train_dataset.column_names)

    # Ensure model is in training mode for Fisher estimation and training
    model.train()

    # Vocabulary check BEFORE training
    tokenizer_vocab = set(processor.tokenizer.get_vocab().keys())
    with open(METADATA_FILE) as f:
        for line in f:
            text = line.strip().split('\t')[1]  # Adjust index if needed
            for char in set(text):
                if char not in tokenizer_vocab:
                    logger.warning(f"Missing character in vocab: {char}")
    model.train()  # Ensure gradients are enabled

    # Data collator for CTC
    data_collator = AudioDataCollatorCTC(processor=processor, padding=True)

    # Metric for evaluation
    wer_metric = evaluate.load("wer")  # FIXED: Use evaluate.load instead of load_metric

    def compute_metrics(pred):
        pred_ids = pred.predictions.argmax(-1)
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Estimate Fisher information for EWC using a subset of Dutch data
    # FIXED: Use DUTCH_CV_ROOT instead of looking in model checkpoint directory
    # FIXED: Use 5000 samples with random sampling for robust Fisher estimation
    try:
        logger.info("Loading Dutch dataset for EWC reference...")
        dutch_metadata = DUTCH_CV_ROOT / "validated.tsv"  # FIXED: Correct path
        dutch_audio_dir = DUTCH_CV_ROOT / "clips"  # FIXED: Correct path
        
        # Get Fisher sample count from config, default to 5000
        fisher_samples = config.get("ewc", {}).get("fisher_samples", 5000)
        logger.info(f"Using {fisher_samples} Dutch samples for Fisher Information estimation")
        
        if dutch_metadata.exists() and dutch_audio_dir.exists():
            # Prepare Dutch dataset for Fisher Information estimation
            # FIXED: Use configurable limit with random sampling
            dutch_dataset = prepare_dataset(
                dutch_metadata, 
                dutch_audio_dir, 
                limit=fisher_samples,
                random_sample=True  # FIXED: Random sampling to avoid bias
            )
            # Preprocess dutch_dataset same way (input_values / labels)
            dutch_dataset = dutch_dataset.map(prepare_examples, batched=True, batch_size=8)
            # Convert to PyTorch DataLoader
            dutch_loader = DataLoader(dutch_dataset, batch_size=8, collate_fn=data_collator)
            # Compute Fisher Information Matrix for EWC
            fisher_dict = compute_fisher_information(model, dutch_loader, device=device)
            old_params = get_model_params(model)
            logger.info("EWC setup complete with Fisher information from Dutch data")
        else:
            logger.warning("Dutch reference data not found for EWC. Proceeding without EWC.")
            logger.warning(f"Expected Dutch data at: {dutch_metadata} and {dutch_audio_dir}")
            fisher_dict = None
            old_params = None
    except Exception as e:
        logger.warning(f"EWC Fisher estimation failed: {e}")
        fisher_dict = None
        old_params = None

    # Training arguments
    # --- Sanitize TRAIN_ARGS types (same approach as pretrain script) ---
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
    bool_keys = {"fp16"}

    for k in int_keys:
        if k in TRAIN_ARGS and isinstance(TRAIN_ARGS[k], str):
            try:
                TRAIN_ARGS[k] = int(TRAIN_ARGS[k])
                logger.info(f"Coerced TRAIN_ARGS['{k}'] from str -> int ({TRAIN_ARGS[k]})")
            except Exception:
                logger.warning(f"Failed to coerce TRAIN_ARGS['{k}']='{TRAIN_ARGS[k]}' to int; leaving as-is")
    for k in float_keys:
        if k in TRAIN_ARGS and isinstance(TRAIN_ARGS[k], str):
            try:
                TRAIN_ARGS[k] = float(TRAIN_ARGS[k])
                logger.info(f"Coerced TRAIN_ARGS['{k}'] from str -> float ({TRAIN_ARGS[k]})")
            except Exception:
                logger.warning(f"Failed to coerce TRAIN_ARGS['{k}']='{TRAIN_ARGS[k]}' to float; leaving as-is")
    for k in bool_keys:
        if k in TRAIN_ARGS and isinstance(TRAIN_ARGS[k], str):
            TRAIN_ARGS[k] = TRAIN_ARGS[k].strip().lower() in ("1", "true", "yes", "y")
            logger.info(f"Coerced TRAIN_ARGS['{k}'] from str -> bool ({TRAIN_ARGS[k]})")

    logger.debug("TRAIN_ARGS after sanitization: %s", TRAIN_ARGS)
    training_args = TrainingArguments(**TRAIN_ARGS)

    # Trainer setup (with EWC if available)
    try:
        eval_dataset = None
        callbacks = []
        if eval_dataset is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

        if fisher_dict is not None and old_params is not None:
            ewc_lambda = config.get("ewc", {}).get("lambda", 0.4)
            logger.info(f"Using EWCTrainer for adaptation with lambda={ewc_lambda}")
            trainer = EWCTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=processor.feature_extractor,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
                ewc_lambda=ewc_lambda,
                fisher_dict=fisher_dict,
                old_params=old_params,
            )
        else:
            logger.info("Using standard Trainer (no EWC).")
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
    except Exception as e:
        logger.error(f"Trainer setup failed: {e}")
        sys.exit(1)

    # Training loop with checkpointing
    try:
        logger.info("Starting adaptation training...")
        train_result = trainer.train()
        logger.info("Adaptation training completed.")
        trainer.save_model(str(OUTPUT_DIR))
        trainer.save_state()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    # Save training logs
    try:
        with open(RESULTS_LOG, "w") as f:
            f.write(str(train_result.metrics))
        logger.info(f"Adaptation metrics saved to {RESULTS_LOG}")
    except Exception as e:
        logger.warning(f"Failed to save adaptation metrics: {e}")

    logger.info("German ASR Adaptation Script Finished.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)