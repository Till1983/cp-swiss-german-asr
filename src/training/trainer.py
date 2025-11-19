import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.integrations import TensorBoardCallback, WandbCallback
from src.config import MODELS_DIR, RESULTS_DIR
import json

logger = logging.getLogger(__name__)

class CatastrophicForgettingCallback(TrainerCallback):
    """
    Custom callback to monitor catastrophic forgetting during adaptation.
    Logs metrics and can trigger actions if forgetting is detected.
    """
    def __init__(self, eval_dataset_pretrain, metric_fn, threshold: float = 0.1):
        """
        Args:
            eval_dataset_pretrain: Dataset from pre-training domain (e.g., Dutch)
            metric_fn: Function to compute metric (e.g., WER) on eval_dataset_pretrain
            threshold: If metric worsens by more than threshold, log warning
        """
        self.eval_dataset_pretrain = eval_dataset_pretrain
        self.metric_fn = metric_fn
        self.threshold = threshold
        self.best_metric = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.eval_dataset_pretrain is not None and self.metric_fn is not None:
            metric = self.metric_fn(self.eval_dataset_pretrain)
            if self.best_metric is None:
                self.best_metric = metric
            elif metric - self.best_metric > self.threshold:
                logger.warning(f"Catastrophic forgetting detected: metric worsened by {metric - self.best_metric:.3f}")
            # Optionally, log to TensorBoard/WandB here

class ProjectTrainer(Trainer):
    """
    Hugging Face Trainer wrapper with project-specific customizations:
    - Logging to TensorBoard or Weights & Biases
    - Checkpoint saving with meaningful names
    - Early stopping based on validation loss
    - Custom callbacks for catastrophic forgetting prevention
    """

    def __init__(
        self,
        *args,
        run_name: Optional[str] = None,
        use_wandb: bool = False,
        catastrophic_forgetting_callback: Optional[CatastrophicForgettingCallback] = None,
        **kwargs
    ):
        """
        Args:
            run_name: Name for the training run (used in logging/checkpoint naming)
            use_wandb: If True, log to Weights & Biases
            catastrophic_forgetting_callback: Custom callback for catastrophic forgetting
        """
        self.run_name = run_name or "asr_training"
        self.use_wandb = use_wandb
        self.catastrophic_forgetting_callback = catastrophic_forgetting_callback

        # Set output and logging directories
        output_dir = kwargs.get("output_dir", MODELS_DIR / "checkpoints" / self.run_name)
        kwargs["output_dir"] = str(output_dir)
        kwargs["logging_dir"] = str(RESULTS_DIR / "logs" / self.run_name)

        # Add TensorBoard/WandB callbacks
        callbacks = kwargs.get("callbacks", [])
        callbacks.append(TensorBoardCallback)
        if self.use_wandb:
            callbacks.append(WandbCallback)
        # Early stopping
        callbacks.append(EarlyStoppingCallback)
        # Catastrophic forgetting
        if self.catastrophic_forgetting_callback:
            callbacks.append(self.catastrophic_forgetting_callback)
        kwargs["callbacks"] = callbacks

        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Save checkpoint with meaningful name including epoch, step, and val_loss.
        """
        checkpoint_dir = Path(self.args.output_dir)
        epoch = int(self.state.epoch or 0)
        step = self.state.global_step
        val_loss = metrics.get("eval_loss") if metrics else None

        ckpt_name = f"checkpoint-epoch{epoch}-step{step}"
        if val_loss is not None:
            ckpt_name += f"-valloss{val_loss:.4f}"
        ckpt_path = checkpoint_dir / ckpt_name

        logger.info(f"Saving checkpoint to {ckpt_path}")
        os.makedirs(ckpt_path, exist_ok=True)
        self.save_model(str(ckpt_path))
        self.state.save_to_json(str(ckpt_path / "trainer_state.json"))

    def log_metrics(self, split: str, metrics: Dict[str, Any]):
        """
        Log metrics to TensorBoard/WandB and save to file.
        """
        super().log_metrics(split, metrics)
        # Save metrics to file for reproducibility
        metrics_path = Path(self.args.logging_dir) / f"{split}_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        Evaluate and log metrics, including catastrophic forgetting if callback is set.
        """
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        self.log_metrics("eval", metrics)
        return metrics

    def train(self, *args, **kwargs):
        """
        Run training loop with logging and checkpointing.
        """
        logger.info(f"Starting training: {self.run_name}")
        train_output = super().train(*args, **kwargs)
        logger.info("Training complete.")
        return train_output

# Example usage:
# from transformers import TrainingArguments
# training_args = TrainingArguments(
#     output_dir=str(MODELS_DIR / "checkpoints" / "dutch_pretrain"),
#     evaluation_strategy="steps",
#     save_strategy="steps",
#     save_steps=500,
#     eval_steps=500,
#     logging_dir=str(RESULTS_DIR / "logs" / "dutch_pretrain"),
#     logging_steps=100,
#     early_stopping_patience=3,
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
#     report_to=["tensorboard"],
# )
# trainer = ProjectTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     run_name="dutch_pretrain",
#     use_wandb=True,
#     catastrophic_forgetting_callback=CatastrophicForgettingCallback(...),
# )
# trainer.train()