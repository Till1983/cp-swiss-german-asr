import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime
import torch
import torchaudio
from datasets import Dataset, Audio
from dataclasses import dataclass
import pandas as pd
import numpy as np

"""
Dutch Pre-training Script for Wav2Vec2 Model

This script handles the pre-training phase of the German Wav2Vec2 model on Dutch Common Voice data.
Pre-training helps the model adapt to a similar low-resource language before fine-tuning on Swiss German.

Usage:
    python scripts/train_dutch_pretrain.py --model facebook/wav2vec2-large-xlsr-53-german \
                                           --train-data data/metadata/dutch/train.tsv \
                                           --output-dir models/pretrained/wav2vec2-dutch \
                                           --epochs 5 \
                                           --batch-size 8
"""


from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
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
        logging.FileHandler(RESULTS_DIR / "logs" / f"dutch_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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


def load_dutch_dataset(metadata_path: Path) -> Dataset:
    """
    Load Dutch Common Voice dataset from TSV metadata file.
    
    Args:
        metadata_path: Path to train.tsv file
        
    Returns:
        HuggingFace Dataset object
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
        ValueError: If required columns are missing
    """
    logger.info(f"Loading Dutch dataset from {metadata_path}")
    
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
    # For pre-training, we primarily monitor loss
    # More sophisticated metrics (WER, CER) come during fine-tuning
    return {}


def create_vocabulary(dataset: Dataset, output_path: Path) -> Dict:
    """
    Create vocabulary from dataset transcriptions.
    
    Args:
        dataset: HuggingFace Dataset with 'sentence' column
        output_path: Path to save vocab.json
        
    Returns:
        Vocabulary dictionary
    """
    logger.info("Creating vocabulary from Dutch transcriptions...")
    
    # Extract all unique characters
    all_text = " ".join(dataset["sentence"])
    vocab = list(set(all_text))
    
    # Add special tokens
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab))}
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    
    # Save vocabulary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as vocab_file:
        json.dump(vocab_dict, vocab_file, ensure_ascii=False, indent=2)
    
    logger.info(f"Created vocabulary with {len(vocab_dict)} tokens")
    logger.info(f"Saved vocabulary to {output_path}")
    
    return vocab_dict


def train_dutch_pretrain(
    model_name: str,
    train_data_path: Path,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    warmup_steps: int = 500,
    eval_steps: int = 500,
    save_steps: int = 1000,
    logging_steps: int = 100,
    gradient_accumulation_steps: int = 2,
    fp16: bool = True,
):
    """
    Pre-train Wav2Vec2 model on Dutch Common Voice data.
    
    Args:
        model_name: HuggingFace model identifier
        train_data_path: Path to Dutch training metadata TSV
        output_dir: Directory to save checkpoints and results
        epochs: Number of training epochs
        batch_size: Training batch size (per device)
        learning_rate: Learning rate for optimizer
        warmup_steps: Number of warmup steps for learning rate scheduler
        eval_steps: Evaluation frequency
        save_steps: Checkpoint save frequency
        logging_steps: Logging frequency
        gradient_accumulation_steps: Steps to accumulate gradients
        fp16: Whether to use mixed precision training
        
    Raises:
        RuntimeError: If training fails
    """
    logger.info("=" * 80)
    logger.info("DUTCH PRE-TRAINING STARTING")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Training data: {train_data_path}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Loading Dataset")
        logger.info("=" * 80)
        train_dataset = load_dutch_dataset(train_data_path)
        
        # Create vocabulary
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Creating Vocabulary")
        logger.info("=" * 80)
        vocab_path = output_dir / "vocab.json"
        vocab_dict = create_vocabulary(train_dataset, vocab_path)
        
        # Initialize tokenizer
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Initializing Tokenizer")
        logger.info("=" * 80)
        tokenizer = Wav2Vec2CTCTokenizer(
            str(vocab_path),
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|"
        )
        
        # Initialize feature extractor
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )
        
        # Create processor
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer
        )
        
        # Save processor
        processor.save_pretrained(output_dir)
        logger.info(f"Saved processor to {output_dir}")
        
        # Load model
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Loading Model")
        logger.info("=" * 80)
        model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            vocab_size=len(vocab_dict),
            pad_token_id=processor.tokenizer.pad_token_id,
            use_safetensors=True
        )
        
        # Freeze feature encoder (optional - for faster training)
        model.freeze_feature_encoder()
        logger.info("Froze feature encoder for faster training")
        
        # Prepare dataset
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Preparing Dataset")
        logger.info("=" * 80)
        train_dataset = train_dataset.map(
            lambda batch: prepare_dataset(batch, processor),
            remove_columns=train_dataset.column_names,
            num_proc=4,
            desc="Preparing training data"
        )
        
        # Data collator
        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
        
        # Training arguments
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Configuring Training")
        logger.info("=" * 80)
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            group_by_length=True,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            evaluation_strategy="steps",
            num_train_epochs=epochs,
            fp16=fp16 and torch.cuda.is_available(),
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            save_total_limit=2,
            push_to_hub=False,
            load_best_model_at_end=False,
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
        logger.info(f"  FP16: {training_args.fp16}")
        logger.info(f"  Device: {training_args.device}")
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            tokenizer=processor.feature_extractor,
        )
        
        # Train
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: Training Model")
        logger.info("=" * 80)
        logger.info("Starting training... This may take several hours.")
        
        train_result = trainer.train()
        
        # Save final model
        logger.info("\n" + "=" * 80)
        logger.info("STEP 8: Saving Model")
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
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
                "train_samples": len(train_dataset),
            },
            "final_metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        
        config_path = output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_summary, f, indent=2)
        
        logger.info(f"\nSaved training configuration to {config_path}")
        
        # Log final metrics
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Final training loss: {metrics.get('train_loss', 'N/A'):.4f}")
        logger.info(f"Model saved to: {output_dir}")
        logger.info(f"Total training time: {metrics.get('train_runtime', 0):.2f} seconds")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error("TRAINING FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise RuntimeError(f"Dutch pre-training failed: {e}") from e


def main():
    """Main entry point for Dutch pre-training script."""
    parser = argparse.ArgumentParser(
        description="Pre-train Wav2Vec2 model on Dutch Common Voice data"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/wav2vec2-large-xlsr-53-german",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=DATA_DIR / "metadata" / "dutch" / "train.tsv",
        help="Path to Dutch training metadata TSV"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR / "pretrained" / "wav2vec2-dutch",
        help="Output directory for model checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
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
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
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
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.train_data.exists():
        logger.error(f"Training data not found: {args.train_data}")
        sys.exit(1)
    
    # Run training
    train_dutch_pretrain(
        model_name=args.model,
        train_data_path=args.train_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=not args.no_fp16,
    )


if __name__ == "__main__":
    main()