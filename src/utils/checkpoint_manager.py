import os
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Any
from src.config import MODELS_DIR, RESULTS_DIR

"""
Checkpoint Manager Utility for Swiss German ASR Project.

Supports:
- Registry and management of model checkpoints across all training phases/sub-tasks.
- Selection of best checkpoints based on validation metrics (e.g., WER, CER, BLEU).
- Conversion between model formats (e.g., PyTorch, SafeTensors, HuggingFace, ONNX) if required.
- Persistent registry of all checkpoints and their associated metrics.
- Uses environment-specific paths from src/config.py.
- Interacts with filesystem via pathlib.Path.

Usage:

    manager = CheckpointManager(task_name="asr_finetune")
    manager.register_checkpoint(...)
    best_ckpt = manager.get_best_checkpoint(metric="wer", mode="min")
    manager.convert_checkpoint(...)
    registry = manager.get_registry()
"""


CHECKPOINT_REGISTRY_FILENAME = "checkpoint_registry.json"

class CheckpointManager:
    """
    Manages model checkpoints for training, validation, and conversion.
    """

    def __init__(self, task_name: str):
        """
        Initialize CheckpointManager for a specific training sub-task.

        Args:
            task_name: Name of the fine-tuning sub-task (e.g., 'asr_finetune', 'accent_adapt', 'domain_adapt')
        """
        self.task_name = task_name
        self.checkpoint_dir = MODELS_DIR / "fine_tuned" / task_name
        self.registry_path = self.checkpoint_dir / CHECKPOINT_REGISTRY_FILENAME
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load checkpoint registry from disk, or initialize if missing."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        """Persist registry to disk."""
        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)

    def register_checkpoint(
        self,
        checkpoint_path: Path,
        metrics: Dict[str, float],
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None
    ):
        """
        Register a new checkpoint and its metrics.

        Args:
            checkpoint_path: Path to checkpoint file (absolute or relative to checkpoint_dir)
            metrics: Dict of validation metrics (e.g., {'wer': 0.21, 'cer': 0.09})
            epoch: Training epoch (optional)
            step: Training step (optional)
            extra: Additional metadata (optional)
        """
        ckpt_name = checkpoint_path.name
        entry = {
            "path": str(checkpoint_path.resolve()),
            "metrics": metrics,
            "epoch": epoch,
            "step": step,
            "extra": extra or {}
        }
        self._registry[ckpt_name] = entry
        self._save_registry()

    def get_best_checkpoint(self, metric: str = "wer", mode: str = "min") -> Optional[Dict[str, Any]]:
        """
        Select the best checkpoint based on a validation metric.

        Args:
            metric: Metric to use for selection (e.g., 'wer', 'cer', 'bleu')
            mode: 'min' for lowest value (WER/CER), 'max' for highest (BLEU)

        Returns:
            Registry entry for best checkpoint, or None if none found.
        """
        candidates = [
            (ckpt, entry)
            for ckpt, entry in self._registry.items()
            if metric in entry["metrics"]
        ]
        if not candidates:
            return None
        if mode == "min":
            best = min(candidates, key=lambda x: x[1]["metrics"][metric])
        else:
            best = max(candidates, key=lambda x: x[1]["metrics"][metric])
        return best[1]

    def convert_checkpoint(
        self,
        checkpoint_path: Path,
        target_format: str = "safetensors",
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Convert checkpoint to another format if needed.

        Args:
            checkpoint_path: Path to source checkpoint
            target_format: Target format ('safetensors', 'pt', 'onnx')
            output_path: Optional output path (defaults to checkpoint_dir)

        Returns:
            Path to converted checkpoint

        Note:
            - Only basic conversion is supported (copy for .pt/.bin/.safetensors).
            - For ONNX, requires torch.onnx export (model class must be provided externally).
        """
        src = checkpoint_path.resolve()
        ext_map = {
            "safetensors": ".safetensors",
            "pt": ".pt",
            "onnx": ".onnx"
        }
        if target_format not in ext_map:
            raise ValueError(f"Unsupported target format: {target_format}")

        out_path = output_path or (self.checkpoint_dir / (src.stem + ext_map[target_format]))

        if target_format in ["safetensors", "pt"]:
            shutil.copy2(src, out_path)
        elif target_format == "onnx":
            raise NotImplementedError(
                "ONNX export requires model instance and input example. "
                "Use torch.onnx.export externally."
            )
        return out_path

    def get_registry(self) -> Dict[str, Any]:
        """Return the full checkpoint registry."""
        return self._registry

    def list_checkpoints(self) -> List[str]:
        """List all registered checkpoint names."""
        return list(self._registry.keys())

    def get_checkpoint_entry(self, ckpt_name: str) -> Optional[Dict[str, Any]]:
        """Get registry entry for a specific checkpoint."""
        return self._registry.get(ckpt_name)

    def remove_checkpoint(self, ckpt_name: str, delete_file: bool = False):
        """
        Remove a checkpoint from the registry (and optionally delete the file).

        Args:
            ckpt_name: Name of checkpoint file
            delete_file: If True, delete the checkpoint file from disk
        """
        entry = self._registry.pop(ckpt_name, None)
        self._save_registry()
        if delete_file and entry:
            try:
                os.remove(entry["path"])
            except Exception:
                pass

    def update_metrics(self, ckpt_name: str, metrics: Dict[str, float]):
        """
        Update metrics for a registered checkpoint.

        Args:
            ckpt_name: Name of checkpoint file
            metrics: Dict of new metrics
        """
        if ckpt_name in self._registry:
            self._registry[ckpt_name]["metrics"].update(metrics)
            self._save_registry()