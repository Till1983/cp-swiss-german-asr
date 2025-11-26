import os
import sys
import logging

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from datasets import Dataset
import evaluate
import pandas as pd  # For direct TSV reading
from src.data.loader import load_audio  # Only need load_audio now
from src.models.wav2vec2_model import Wav2Vec2Model
from src.config import DUTCH_CV_ROOT, GERMAN_CV_ROOT, MODELS_DIR, RESULTS_DIR
import numpy as np
from torch.utils.data import DataLoader
import yaml
from transformers import Wav2Vec2ForCTC
from src.data.collator import AudioDataCollatorCTC
from tqdm import tqdm

"""
German ASR Adaptation Script with EWC

This script loads the Dutch-pretrained Wav2Vec2 checkpoint, applies elastic weight consolidation (EWC)
to mitigate catastrophic forgetting, and fine-tunes the model on German Common Voice data.

Features:
- Comprehensive progress logging at every stage
- Optimized dataset loading with smart file validation
- Configurable data limits for efficient training
- EWC implementation for catastrophic forgetting prevention

Usage:
    python scripts/train_german_adaptation.py

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
logger.info("Loading configuration from german_adaptation.yml...")
with open("configs/training/german_adaptation.yml", "r") as f:
    config = yaml.safe_load(f)

# Use checkpoint path from config with smart resolution
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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override compute_loss to add EWC penalty."""
        # Call parent with proper parameter handling
        if num_items_in_batch is not None:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        else:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        ewc_loss = self.compute_ewc_loss()
        total_loss = loss + ewc_loss
        
        if return_outputs:
            return total_loss, outputs
        return total_loss

def compute_fisher_information(model, dataloader, device):
    """
    Estimate Fisher information for model parameters using provided dataloader.
    
    Args:
        model: The model to compute Fisher information for
        dataloader: DataLoader containing reference samples (e.g., Dutch data)
        device: Device to run computation on
    
    Returns:
        fisher_dict: Dictionary mapping parameter names to Fisher information
    """
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    logger.info(f"=" * 70)
    logger.info(f"FISHER INFORMATION ESTIMATION (EWC)")
    logger.info(f"=" * 70)
    logger.info(f"Samples: {num_samples}")
    logger.info(f"Batches: {num_batches}")
    logger.info(f"This will take approximately {num_batches * 0.5 / 60:.1f} minutes")
    logger.info(f"=" * 70)
    
    fisher_dict = {}
    model.eval()
    
    # Initialize Fisher dict with zeros
    for name, param in model.named_parameters():
        fisher_dict[name] = torch.zeros_like(param)
    
    # Compute gradients for each batch with progress bar
    logger.info("Computing Fisher information (squared gradients)...")
    for i, batch in enumerate(tqdm(dataloader, desc="Fisher estimation", unit="batch")):
        if i % 50 == 0 and i > 0:
            logger.info(f"  Progress: {i}/{num_batches} batches ({100*i/num_batches:.1f}%)")
        
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
    logger.info("Averaging Fisher information across batches...")
    for name in fisher_dict:
        fisher_dict[name] /= num_batches
    
    logger.info("✅ Fisher information estimation complete!")
    logger.info(f"=" * 70)
    return fisher_dict

def get_model_params(model):
    """
    Get a copy of model parameters for EWC reference.
    """
    return {name: param.clone().detach() for name, param in model.named_parameters()}

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def prepare_dataset(metadata_path, audio_dir, limit=None, random_sample=False, skip_validation=False):
    """
    Prepare German Common Voice dataset for ASR training with optimized loading.

    Args:
        metadata_path: Path to TSV metadata file.
        audio_dir: Directory containing audio files.
        limit: Optional limit on number of samples.
        random_sample: If True and limit is set, sample randomly instead of taking first N.
        skip_validation: If True, skip file existence checks (faster, assumes data integrity).

    Returns:
        Hugging Face Dataset object.
    """
    logger.info(f"=" * 70)
    logger.info(f"DATASET PREPARATION")
    logger.info(f"=" * 70)
    logger.info(f"Metadata: {metadata_path}")
    logger.info(f"Audio dir: {audio_dir}")
    if limit:
        logger.info(f"Limit: {limit} samples")
        logger.info(f"Sampling: {'Random' if random_sample else 'Sequential'}")
    logger.info(f"Validation: {'Skipped' if skip_validation else 'Enabled'}")
    logger.info(f"=" * 70)
    
    # Load metadata with proper handling for large TSV files
    logger.info("Loading metadata from TSV...")
    try:
        # Use low_memory=False to avoid dtype warnings and ensure proper parsing
        df = pd.read_csv(
            str(metadata_path), 
            sep='\t',
            low_memory=False,
            na_values=['', 'NA', 'nan', 'NaN'],
            keep_default_na=True,
            encoding='utf-8',
            quoting=3  # QUOTE_NONE - don't interpret quotes
        )
        logger.info(f"✅ Loaded {len(df):,} rows from metadata")
    except Exception as e:
        logger.error(f"❌ Failed to load metadata: {e}")
        raise

    # Check required columns
    required_columns = {'sentence', 'path'}
    if not required_columns.issubset(df.columns):
        logger.error(f"❌ Metadata missing required columns: {required_columns}")
        raise ValueError("Invalid metadata file format.")

    # Apply sampling if requested
    if limit:
        original_size = len(df)
        if random_sample:
            logger.info(f"Randomly sampling {limit:,} from {original_size:,} samples...")
            df = df.sample(n=min(limit, len(df)), random_state=42)
        else:
            logger.info(f"Taking first {limit:,} from {original_size:,} samples...")
            df = df.head(limit)
        logger.info(f"✅ Dataset size after sampling: {len(df):,} samples")

    # Add absolute audio paths with robust path cleaning
    logger.info("Adding audio file paths...")
    def construct_audio_path(path_value):
        """Construct audio path, handling both relative and absolute paths."""
        # Clean the path value aggressively
        path_value = str(path_value).strip()  # Remove leading/trailing whitespace
        path_value = path_value.replace('\n', '').replace('\r', '')  # Remove newlines
        path_value = path_value.replace('\t', '')  # Remove tabs
        
        if not path_value or path_value == 'nan':
            return None
            
        path_obj = Path(path_value)
        
        # If already absolute, use as-is
        if path_obj.is_absolute():
            return str(path_obj)
        
        # If relative, prepend audio_dir
        # Handle case where path might already include 'clips/'
        if path_value.startswith('clips/'):
            # Remove 'clips/' prefix and add to audio_dir
            filename = path_value[6:]  # Remove 'clips/'
            return str(audio_dir / filename)
        else:
            # Just filename, add to audio_dir
            return str(audio_dir / path_value)
    
    df['audio_path'] = df['path'].apply(construct_audio_path)
    
    # Remove rows with invalid paths
    initial_count = len(df)
    df = df[df['audio_path'].notna()]
    if len(df) < initial_count:
        logger.warning(f"⚠️  Removed {initial_count - len(df)} rows with invalid paths")
    
    logger.info(f"✅ Added audio paths")
    
    # Log a few examples for debugging
    logger.info("Sample audio paths:")
    for i, path in enumerate(df['audio_path'].head(3)):
        logger.info(f"  [{i}] {path}")
        # Verify first few actually exist
        exists = Path(path).exists()
        logger.info(f"      Exists: {exists}")

    # File validation (optional, can be slow for large datasets)
    if not skip_validation:
        logger.info("Validating audio file availability...")
        logger.info("  (Sampling first 100 files for quick check)")
        
        # Sample check: verify first 100 files exist
        sample_size = min(100, len(df))
        sample_paths = df['audio_path'].head(sample_size).tolist()
        missing_in_sample = sum(1 for p in sample_paths if not Path(p).exists())
        
        if missing_in_sample == 0:
            logger.info(f"✅ All {sample_size} sampled files exist - assuming dataset is complete")
        elif missing_in_sample < 10:
            logger.warning(f"⚠️  {missing_in_sample}/{sample_size} sampled files missing - proceeding anyway")
        else:
            logger.warning(f"⚠️  {missing_in_sample}/{sample_size} sampled files missing!")
            logger.info("  Performing full validation (this may take several minutes)...")
            initial_count = len(df)
            
            # Full validation with progress bar
            valid_mask = []
            for path in tqdm(df['audio_path'], desc="Checking files", unit="file"):
                valid_mask.append(Path(path).exists())
            df = df[valid_mask]
            
            removed = initial_count - len(df)
            logger.info(f"✅ Validation complete: removed {removed:,} samples with missing files")
    else:
        logger.info("⚠️  Skipping file validation (assuming data integrity)")

    # Create HuggingFace Dataset
    logger.info("Creating HuggingFace Dataset object...")
    dataset = Dataset.from_pandas(df)
    logger.info(f"✅ Dataset created with {len(dataset):,} samples")

    # Add audio loading function
    logger.info("Mapping audio loading function...")
    def map_audio(batch):
        try:
            batch["audio"] = load_audio(batch["audio_path"])
        except Exception as e:
            logger.warning(f"⚠️  Audio load failed for {batch['audio_path']}: {e}")
            batch["audio"] = None
        return batch

    dataset = dataset.map(map_audio)
    
    # Filter out failed audio loads
    initial_size = len(dataset)
    dataset = dataset.filter(lambda x: x["audio"] is not None)
    failed_loads = initial_size - len(dataset)
    
    if failed_loads > 0:
        logger.warning(f"⚠️  Filtered {failed_loads} samples with failed audio loading")

    logger.info(f"=" * 70)
    logger.info(f"✅ DATASET READY: {len(dataset):,} samples")
    logger.info(f"=" * 70)
    return dataset

# -----------------------------------------------------------------------------
# Main Training Routine
# -----------------------------------------------------------------------------
def main():
    logger.info("=" * 70)
    logger.info("GERMAN ASR ADAPTATION WITH EWC")
    logger.info("=" * 70)
    logger.info("Starting German ASR Adaptation Script")
    logger.info(f"Config: {config}")
    logger.info("=" * 70)

    # Ensure output directories exist
    logger.info("Creating output directories...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Output dir: {OUTPUT_DIR}")
    logger.info(f"✅ Checkpoint dir: {CHECKPOINT_DIR}")

    # Load German training dataset with optimized settings
    logger.info("")
    logger.info("STEP 1: LOADING GERMAN TRAINING DATA")
    logger.info("=" * 70)
    try:
        # Use 50k subset for efficient adaptation (standard practice in transfer learning)
        # Skip file validation for speed (assumes data integrity from Common Voice)
        train_dataset = prepare_dataset(
            METADATA_FILE, 
            AUDIO_DIR,
            limit=150000,           # 150k samples = ~25% of full dataset, sufficient for adaptation
            random_sample=True,    # Random sampling ensures diverse coverage
            skip_validation=False   # Skip slow file checks, trust Common Voice data quality
        )
        logger.info(f"✅ German training dataset loaded successfully")
    except Exception as e:
        logger.error(f"❌ Dataset preparation failed: {e}")
        sys.exit(1)

    # Load pretrained Dutch model and processor
    logger.info("")
    logger.info("STEP 2: LOADING DUTCH PRETRAINED MODEL")
    logger.info("=" * 70)
    try:
        logger.info(f"Loading model from: {MODEL_NAME}")
        model_wrapper = Wav2Vec2Model(model_name=MODEL_NAME)
        model = model_wrapper.model.to(device)
        processor = model_wrapper.processor
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Model type: {type(model).__name__}")
        logger.info(f"   Device: {device}")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        sys.exit(1)

    # Tokenizer sanity check
    logger.info("")
    logger.info("STEP 3: TOKENIZER VALIDATION")
    logger.info("=" * 70)
    try:
        logger.info("Testing tokenizer on sample sentences...")
        sample_count = min(5, len(train_dataset))
        for i, row in enumerate(train_dataset.select(range(sample_count))):
            sent = row.get("sentence", "")
            toks = processor.tokenizer(sent, add_special_tokens=False)
            token_ids = getattr(toks, "input_ids", toks.get("input_ids", None))
            if not token_ids:
                logger.warning(f"⚠️  Sample {i}: Tokenizer produced no tokens")
            else:
                logger.info(f"  Sample {i}: '{sent[:50]}...' -> {len(token_ids)} tokens")
        logger.info(f"✅ Tokenizer validation complete")
    except Exception as e:
        logger.warning(f"⚠️  Tokenizer sanity check failed: {e}")

    # Preprocess dataset
    logger.info("")
    logger.info("STEP 4: DATASET PREPROCESSING")
    logger.info("=" * 70)
    sampling_rate = config.get("data", {}).get("sampling_rate", 16000)
    logger.info(f"Sampling rate: {sampling_rate} Hz")

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

    logger.info("Mapping preprocessing function to dataset...")
    train_dataset = train_dataset.map(prepare_examples, batched=True, batch_size=4)
    logger.info(f"✅ Preprocessing complete")
    logger.info(f"   Dataset columns: {train_dataset.column_names}")

    # Ensure model is in training mode
    model.train()

    # Vocabulary check
    logger.info("")
    logger.info("STEP 5: VOCABULARY CHECK")
    logger.info("=" * 70)
    logger.info("Checking for missing characters in tokenizer vocabulary...")
    tokenizer_vocab = set(processor.tokenizer.get_vocab().keys())
    missing_chars = set()
    
    with open(METADATA_FILE) as f:
        for i, line in enumerate(f):
            if i > 1000:  # Check first 1000 lines
                break
            try:
                text = line.strip().split('\t')[3]  # Sentence column
                for char in set(text):
                    if char not in tokenizer_vocab and char not in missing_chars:
                        missing_chars.add(char)
                        logger.warning(f"⚠️  Missing character in vocab: '{char}'")
            except:
                continue
    
    if missing_chars:
        logger.warning(f"⚠️  Found {len(missing_chars)} missing characters (may affect accuracy)")
    else:
        logger.info(f"✅ All characters in vocabulary")

    # Data collator for CTC
    data_collator = AudioDataCollatorCTC(processor=processor, padding=True)

    # Metric for evaluation
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions.argmax(-1)
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Fisher information estimation for EWC
    logger.info("")
    logger.info("STEP 6: EWC SETUP (FISHER INFORMATION)")
    logger.info("=" * 70)
    
    try:
        logger.info("Loading Dutch reference dataset for EWC...")
        dutch_metadata = DUTCH_CV_ROOT / "validated.tsv"
        dutch_audio_dir = DUTCH_CV_ROOT / "clips"
        
        # Get Fisher sample count from config, default to 5000
        fisher_samples = config.get("ewc", {}).get("fisher_samples", 5000)
        logger.info(f"Target: {fisher_samples} Dutch samples for Fisher estimation")
        
        if dutch_metadata.exists() and dutch_audio_dir.exists():
            logger.info(f"✅ Dutch data found at {dutch_metadata}")
            
            # Prepare Dutch dataset for Fisher Information estimation
            dutch_dataset = prepare_dataset(
                dutch_metadata, 
                dutch_audio_dir, 
                limit=fisher_samples,
                random_sample=True,
                skip_validation=True  # Skip validation for speed
            )
            
            # Preprocess Dutch dataset
            logger.info("Preprocessing Dutch dataset for Fisher estimation...")
            dutch_dataset = dutch_dataset.map(prepare_examples, batched=True, batch_size=8)
            
            # Convert to PyTorch DataLoader
            logger.info("Creating DataLoader for Fisher computation...")
            dutch_loader = DataLoader(dutch_dataset, batch_size=8, collate_fn=data_collator)
            logger.info(f"✅ DataLoader ready: {len(dutch_loader)} batches")
            
            # Compute Fisher Information Matrix for EWC
            fisher_dict = compute_fisher_information(model, dutch_loader, device=device)
            old_params = get_model_params(model)
            logger.info("✅ EWC setup complete with Fisher information from Dutch data")
        else:
            logger.warning("❌ Dutch reference data not found for EWC")
            logger.warning(f"   Expected: {dutch_metadata}")
            logger.warning(f"   Expected: {dutch_audio_dir}")
            logger.warning("   Proceeding WITHOUT EWC (standard fine-tuning)")
            fisher_dict = None
            old_params = None
    except Exception as e:
        logger.warning(f"❌ EWC Fisher estimation failed: {e}")
        logger.warning("   Proceeding WITHOUT EWC (standard fine-tuning)")
        fisher_dict = None
        old_params = None

    # Training arguments
    logger.info("")
    logger.info("STEP 7: TRAINING SETUP")
    logger.info("=" * 70)
    
    # Sanitize TRAIN_ARGS types
    logger.info("Sanitizing training arguments...")
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
                logger.info(f"  Coerced '{k}': {TRAIN_ARGS[k]} (int)")
            except Exception:
                logger.warning(f"  Failed to coerce '{k}': {TRAIN_ARGS[k]}")
                
    for k in float_keys:
        if k in TRAIN_ARGS and isinstance(TRAIN_ARGS[k], str):
            try:
                TRAIN_ARGS[k] = float(TRAIN_ARGS[k])
                logger.info(f"  Coerced '{k}': {TRAIN_ARGS[k]} (float)")
            except Exception:
                logger.warning(f"  Failed to coerce '{k}': {TRAIN_ARGS[k]}")
                
    for k in bool_keys:
        if k in TRAIN_ARGS and isinstance(TRAIN_ARGS[k], str):
            TRAIN_ARGS[k] = TRAIN_ARGS[k].strip().lower() in ("1", "true", "yes", "y")
            logger.info(f"  Coerced '{k}': {TRAIN_ARGS[k]} (bool)")

    logger.info("Creating TrainingArguments...")
    training_args = TrainingArguments(**TRAIN_ARGS)
    logger.info(f"✅ Training arguments created")
    logger.info(f"   Learning rate: {training_args.learning_rate}")
    logger.info(f"   Epochs: {training_args.num_train_epochs}")
    logger.info(f"   Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"   FP16: {training_args.fp16}")

    # Trainer setup (with EWC if available)
    logger.info("")
    logger.info("STEP 8: TRAINER INITIALIZATION")
    logger.info("=" * 70)
    
    try:
        eval_dataset = None
        callbacks = []
        if eval_dataset is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

        if fisher_dict is not None and old_params is not None:
            ewc_lambda = config.get("ewc", {}).get("lambda", 0.4)
            logger.info(f"✅ Using EWCTrainer (catastrophic forgetting prevention)")
            logger.info(f"   EWC lambda: {ewc_lambda}")
            logger.info(f"   Fisher samples: {fisher_samples}")
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
            logger.info("⚠️  Using standard Trainer (no EWC)")
            logger.info("   Training will proceed without catastrophic forgetting prevention")
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
        logger.error(f"❌ Trainer setup failed: {e}")
        sys.exit(1)

    # Training loop
    logger.info("")
    logger.info("STEP 9: TRAINING EXECUTION")
    logger.info("=" * 70)
    logger.info("🚀 STARTING GERMAN ADAPTATION TRAINING")
    logger.info("=" * 70)
    
    try:
        train_result = trainer.train()
        logger.info("=" * 70)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        
        # Save model
        logger.info(f"Saving adapted model to {OUTPUT_DIR}...")
        trainer.save_model(str(OUTPUT_DIR))
        trainer.save_state()
        logger.info(f"✅ Model saved")
        
        # Log final metrics
        logger.info("")
        logger.info("FINAL TRAINING METRICS")
        logger.info("=" * 70)
        for key, value in train_result.metrics.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error(f"   Stack trace:", exc_info=True)
        sys.exit(1)

    # Save training logs
    try:
        RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_LOG, "w") as f:
            f.write(str(train_result.metrics))
        logger.info(f"✅ Metrics saved to {RESULTS_LOG}")
    except Exception as e:
        logger.warning(f"⚠️  Failed to save metrics: {e}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("🎉 GERMAN ASR ADAPTATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Adapted model: {OUTPUT_DIR}")
    logger.info(f"Training logs: {RESULTS_LOG}")
    logger.info(f"Next step: Evaluation on Swiss German dialects")
    logger.info("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("=" * 70)
        logger.warning("⚠️  TRAINING INTERRUPTED BY USER")
        logger.warning("=" * 70)
        sys.exit(1)
    except Exception as e:
        logger.critical("")
        logger.critical("=" * 70)
        logger.critical(f"❌ FATAL ERROR: {e}")
        logger.critical("=" * 70)
        logger.critical("Stack trace:", exc_info=True)
        sys.exit(1)