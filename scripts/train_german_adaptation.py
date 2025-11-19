import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import torch
import pandas as pd
import numpy as np
from datasets import Dataset, Audio
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.loader import load_swiss_german_metadata

# filepath: /Users/tillermold/Desktop/CODE/Synthesis Semester/Capstone_Project/cp-swiss-german-asr/scripts/train_german_adaptation.py
"""
German Adaptation Script with Elastic Weight Consolidation (EWC)

This script loads the Dutch-pretrained German Wav2Vec2 model and fine-tunes it on German Common Voice data
while using Elastic Weight Consolidation to prevent catastrophic forgetting of German knowledge.

EWC helps preserve important weights learned during Dutch pre-training by adding a quadratic penalty
to the loss function for parameters that were important for the Dutch task.

Usage:
    python scripts/train_german_adaptation.py --dutch-checkpoint models/pretrained/wav2vec2-dutch \
                                               --train-data data/metadata/german/train.tsv \
                                               --val-data data/metadata/german/validation.tsv \
                                               --output-dir models/pretrained/wav2vec2-german-adapted \
                                               --epochs 3 \
                                               --ewc-lambda 0.4
"""


import torch.nn.functional as F

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# Import project modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import (
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    ENVIRONMENT,
    get_config_summary,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            RESULTS_DIR / "logs" / f"german_adaptation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    
    Args:
        processor: Wav2Vec2Processor for feature extraction and tokenization
        padding: Whether to pad inputs to the longest sequence in the batch
    """
    processor: Wav2Vec2Processor
    padding: bool = True

    def __call__(self, features):
        # Split inputs and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input features
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad labels
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch


class EWCTrainer(Trainer):
    """
    Custom Trainer that implements Elastic Weight Consolidation (EWC).
    
    EWC adds a regularization term to the loss that penalizes changes to parameters
    that were important for the previous task (Dutch pre-training).
    """
    
    def __init__(self, *args, ewc_lambda: float = 0.4, fisher_matrix: Dict = None, 
                 optimal_params: Dict = None, **kwargs):
        """
        Initialize EWC Trainer.
        
        Args:
            ewc_lambda: Importance weight for EWC penalty (higher = more preservation)
            fisher_matrix: Fisher information matrix from previous task
            optimal_params: Optimal parameters from previous task
        """
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.fisher_matrix = fisher_matrix or {}
        self.optimal_params = optimal_params or {}
        
        logger.info(f"EWC Trainer initialized with lambda={ewc_lambda}")
        logger.info(f"Fisher matrix contains {len(self.fisher_matrix)} parameters")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss with EWC penalty.
        
        Args:
            model: The model being trained
            inputs: Input batch
            return_outputs: Whether to return model outputs
            
        Returns:
            Loss tensor (and optionally outputs)
        """
        # Standard CTC loss
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Add EWC penalty
        if self.fisher_matrix and self.optimal_params:
            ewc_loss = 0.0
            for name, param in model.named_parameters():
                if name in self.fisher_matrix:
                    fisher = self.fisher_matrix[name]
                    optimal = self.optimal_params[name]
                    ewc_loss += (fisher * (param - optimal) ** 2).sum()
            
            loss = loss + (self.ewc_lambda / 2) * ewc_loss
        
        return (loss, outputs) if return_outputs else loss


def compute_fisher_information(
    model: Wav2Vec2ForCTC,
    dataloader: DataLoader,
    device: str,
    num_samples: int = 1000
) -> Tuple[Dict, Dict]:
    """
    Compute Fisher Information Matrix for EWC.
    
    The Fisher matrix approximates the importance of each parameter for the current task.
    Parameters with high Fisher values should be preserved during adaptation.
    
    Args:
        model: Trained model from previous task (Dutch)
        dataloader: DataLoader for Dutch validation data
        device: Device to run computation on
        num_samples: Number of samples to use for Fisher estimation
        
    Returns:
        Tuple of (fisher_matrix, optimal_params) dictionaries
    """
    logger.info("Computing Fisher Information Matrix...")
    logger.info(f"Using {num_samples} samples for Fisher estimation")
    
    model.eval()
    fisher_matrix = defaultdict(float)
    optimal_params = {}
    
    # Store optimal parameters (weights from Dutch pre-training)
    for name, param in model.named_parameters():
        optimal_params[name] = param.data.clone()
    
    # Compute Fisher matrix using empirical Fisher (diagonal approximation)
    samples_processed = 0
    
    with tqdm(total=min(num_samples, len(dataloader)), desc="Computing Fisher") as pbar:
        for batch in dataloader:
            if samples_processed >= num_samples:
                break
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass to get gradients
            model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients (Fisher approximation)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_matrix[name] += param.grad.data ** 2
            
            samples_processed += batch['input_values'].size(0)
            pbar.update(1)
    
    # Normalize Fisher matrix
    for name in fisher_matrix:
        fisher_matrix[name] /= samples_processed
    
    logger.info(f"Fisher Information Matrix computed using {samples_processed} samples")
    logger.info(f"Parameters tracked: {len(fisher_matrix)}")
    
    return dict(fisher_matrix), optimal_params


def load_german_dataset(metadata_path: Path) -> Dataset:
    """
    Load German Common Voice dataset from TSV metadata file.
    
    Args:
        metadata_path: Path to train.tsv or validation.tsv file
        
    Returns:
        HuggingFace Dataset object
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
        ValueError: If required columns are missing
    """
    logger.info(f"Loading German dataset from {metadata_path}")
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Load TSV
    try:
        df = pd.read_csv(metadata_path, sep='\t')
        logger.info(f"Loaded {len(df)} samples from metadata")
    except Exception as e:
        raise ValueError(f"Failed to read metadata file: {e}") from e
    
    # Validate required columns
    required_cols = ['audio_path', 'sentence']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata: {missing_cols}")
    
    # Convert to absolute paths
    df['audio_path'] = df['audio_path'].apply(
        lambda x: str(DATA_DIR.parent / x) if not Path(x).is_absolute() else x
    )
    
    # Filter out missing audio files
    original_count = len(df)
    df = df[df['audio_path'].apply(lambda x: Path(x).exists())]
    filtered_count = original_count - len(df)
    if filtered_count > 0:
        logger.warning(f"Filtered out {filtered_count} samples with missing audio files")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_pandas(df[['audio_path', 'sentence']])
    
    # Cast audio column to Audio feature
    dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))
    dataset = dataset.rename_column("audio_path", "audio")
    
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    return dataset


def prepare_dataset(batch, processor):
    """
    Prepare a batch of data for training.
    
    Args:
        batch: Batch from HuggingFace dataset
        processor: Wav2Vec2Processor
        
    Returns:
        Processed batch with input_values and labels
    """
    # Load and process audio
    audio = batch["audio"]
    
    # Extract input features
    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    
    # Encode text labels
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    
    return batch


def compute_metrics(pred):
    """
    Compute metrics during evaluation.
    
    Args:
        pred: Predictions from model
        
    Returns:
        Dictionary of metrics
    """
    # For adaptation, we primarily monitor loss
    # Detailed metrics (WER, CER) will be computed in separate evaluation
    return {}


def train_german_adaptation(
    dutch_checkpoint: Path,
    train_data_path: Path,
    val_data_path: Optional[Path],
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    ewc_lambda: float = 0.4,
    warmup_steps: int = 300,
    eval_steps: int = 500,
    save_steps: int = 1000,
    logging_steps: int = 100,
    gradient_accumulation_steps: int = 2,
    fp16: bool = True,
    compute_fisher: bool = True,
    fisher_samples: int = 1000,
):
    """
    Fine-tune Dutch-pretrained model on German data with EWC.
    
    Args:
        dutch_checkpoint: Path to Dutch pretrained model checkpoint
        train_data_path: Path to German training metadata TSV
        val_data_path: Optional path to German validation metadata TSV
        output_dir: Directory to save checkpoints and results
        epochs: Number of training epochs
        batch_size: Training batch size (per device)
        learning_rate: Learning rate for optimizer (lower than pre-training)
        ewc_lambda: EWC regularization strength (0 = no EWC, higher = more preservation)
        warmup_steps: Number of warmup steps for learning rate scheduler
        eval_steps: Evaluation frequency
        save_steps: Checkpoint save frequency
        logging_steps: Logging frequency
        gradient_accumulation_steps: Steps to accumulate gradients
        fp16: Whether to use mixed precision training
        compute_fisher: Whether to compute Fisher matrix for EWC
        fisher_samples: Number of samples for Fisher computation
        
    Raises:
        RuntimeError: If training fails
    """
    logger.info("=" * 80)
    logger.info("GERMAN ADAPTATION WITH EWC STARTING")
    logger.info("=" * 80)
    logger.info(f"Dutch checkpoint: {dutch_checkpoint}")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"EWC lambda: {ewc_lambda}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Validate Dutch checkpoint
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Loading Dutch Checkpoint")
        logger.info("=" * 80)
        
        if not dutch_checkpoint.exists():
            raise FileNotFoundError(f"Dutch checkpoint not found: {dutch_checkpoint}")
        
        # Load processor
        processor = Wav2Vec2Processor.from_pretrained(str(dutch_checkpoint))
        logger.info("Loaded processor from Dutch checkpoint")
        
        # Load model
        model = Wav2Vec2ForCTC.from_pretrained(
            str(dutch_checkpoint),
            use_safetensors=True
        )
        logger.info("Loaded model from Dutch checkpoint")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        model.to(device)
        logger.info(f"Model moved to device: {device}")
        
        # Load datasets
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Loading German Datasets")
        logger.info("=" * 80)
        
        train_dataset = load_german_dataset(train_data_path)
        
        val_dataset = None
        if val_data_path and val_data_path.exists():
            val_dataset = load_german_dataset(val_data_path)
            logger.info(f"Loaded validation dataset with {len(val_dataset)} samples")
        else:
            logger.warning("No validation dataset provided")
        
        # Compute Fisher Information Matrix for EWC
        fisher_matrix = {}
        optimal_params = {}
        
        if compute_fisher and ewc_lambda > 0:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3: Computing Fisher Information Matrix")
            logger.info("=" * 80)
            
            # Use validation set for Fisher computation if available, else use train set
            fisher_dataset = val_dataset if val_dataset else train_dataset
            
            # Prepare dataset for Fisher computation
            fisher_dataset_processed = fisher_dataset.map(
                lambda batch: prepare_dataset(batch, processor),
                remove_columns=fisher_dataset.column_names,
                num_proc=4,
                desc="Preparing Fisher dataset"
            )
            
            # Create dataloader
            data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
            fisher_dataloader = DataLoader(
                fisher_dataset_processed,
                batch_size=batch_size,
                collate_fn=data_collator,
                shuffle=False
            )
            
            fisher_matrix, optimal_params = compute_fisher_information(
                model=model,
                dataloader=fisher_dataloader,
                device=device,
                num_samples=fisher_samples
            )
            
            logger.info(f"Fisher computation complete: {len(fisher_matrix)} parameters tracked")
        else:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3: Skipping Fisher Computation (EWC disabled)")
            logger.info("=" * 80)
        
        # Prepare training dataset
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Preparing Datasets for Training")
        logger.info("=" * 80)
        
        train_dataset = train_dataset.map(
            lambda batch: prepare_dataset(batch, processor),
            remove_columns=train_dataset.column_names,
            num_proc=4,
            desc="Preparing training data"
        )
        
        if val_dataset:
            val_dataset = val_dataset.map(
                lambda batch: prepare_dataset(batch, processor),
                remove_columns=val_dataset.column_names,
                num_proc=4,
                desc="Preparing validation data"
            )
        
        # Data collator
        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
        
        # Training arguments
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Configuring Training")
        logger.info("=" * 80)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            group_by_length=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            evaluation_strategy="steps" if val_dataset else "no",
            num_train_epochs=epochs,
            fp16=True if torch.cuda.is_available() or torch.backends.mps.is_available() else False,
            save_steps=save_steps,
            eval_steps=eval_steps if val_dataset else None,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            save_total_limit=3,
            push_to_hub=False,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="loss",
            greater_is_better=False,
            report_to=["tensorboard"],
            logging_dir=str(RESULTS_DIR / "logs" / "tensorboard"),
        )
        
        logger.info(f"Training configuration:")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  EWC lambda: {ewc_lambda}")
        logger.info(f"  FP16: {training_args.fp16}")
        logger.info(f"  Device: {training_args.device}")
        
        # Initialize EWC Trainer
        trainer = EWCTrainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=processor.feature_extractor,
            ewc_lambda=ewc_lambda,
            fisher_matrix=fisher_matrix,
            optimal_params=optimal_params,
        )
        
        # Add early stopping if validation set is provided
        if val_dataset:
            trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
            logger.info("Added early stopping callback (patience=3)")
        
        # Train
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Training Model")
        logger.info("=" * 80)
        logger.info("Starting German adaptation training with EWC...")
        
        train_result = trainer.train()
        
        # Save final model
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: Saving Model")
        logger.info("=" * 80)
        
        trainer.save_model()
        processor.save_pretrained(output_dir)
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Save configuration summary
        config_summary = {
            **get_config_summary(),
            "training_params": {
                "dutch_checkpoint": str(dutch_checkpoint),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "ewc_lambda": ewc_lambda,
                "warmup_steps": warmup_steps,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset) if val_dataset else 0,
                "fisher_computed": compute_fisher,
                "fisher_samples": fisher_samples if compute_fisher else 0,
            },
            "final_metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        
        config_path = output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_summary, f, indent=2)
        
        logger.info(f"Saved training configuration to {config_path}")
        
        # Log final metrics
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Final training loss: {metrics.get('train_loss', 'N/A'):.4f}")
        if val_dataset:
            logger.info(f"Final validation loss: {metrics.get('eval_loss', 'N/A'):.4f}")
        logger.info(f"Model saved to: {output_dir}")
        logger.info(f"Total training time: {metrics.get('train_runtime', 0):.2f} seconds")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error("TRAINING FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise RuntimeError(f"German adaptation training failed: {e}") from e


def main():
    """Main entry point for German adaptation script."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Dutch-pretrained model on German with EWC"
    )
    
    parser.add_argument(
        "--dutch-checkpoint",
        type=Path,
        default=MODELS_DIR / "pretrained" / "wav2vec2-dutch",
        help="Path to Dutch pretrained model checkpoint"
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=DATA_DIR / "metadata" / "german" / "train.tsv",
        help="Path to German training metadata TSV"
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=DATA_DIR / "metadata" / "german" / "validation.tsv",
        help="Path to German validation metadata TSV"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR / "pretrained" / "wav2vec2-german-adapted",
        help="Output directory for adapted model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (lower than pre-training)"
    )
    parser.add_argument(
        "--ewc-lambda",
        type=float,
        default=0.4,
        help="EWC regularization strength (0=disabled)"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=300,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable mixed precision training"
    )
    parser.add_argument(
        "--no-fisher",
        action="store_true",
        help="Skip Fisher matrix computation (disables EWC)"
    )
    parser.add_argument(
        "--fisher-samples",
        type=int,
        default=1000,
        help="Number of samples for Fisher computation"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.dutch_checkpoint.exists():
        logger.error(f"Dutch checkpoint not found: {args.dutch_checkpoint}")
        sys.exit(1)
    
    if not args.train_data.exists():
        logger.error(f"Training data not found: {args.train_data}")
        sys.exit(1)
    
    # Run training
    train_german_adaptation(
        dutch_checkpoint=args.dutch_checkpoint,
        train_data_path=args.train_data,
        val_data_path=args.val_data if args.val_data.exists() else None,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ewc_lambda=args.ewc_lambda,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=not args.no_fp16,
        compute_fisher=not args.no_fisher,
        fisher_samples=args.fisher_samples,
    )


if __name__ == "__main__":
    main()