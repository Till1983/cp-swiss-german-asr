"""Unit tests for checkpoint manager module."""
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


# Patch the config import before importing CheckpointManager
@pytest.fixture(autouse=True)
def mock_config(temp_dir):
    """Mock the config module with temp directories."""
    with patch('src.utils.checkpoint_manager.MODELS_DIR', temp_dir / "models"), \
         patch('src.utils.checkpoint_manager.RESULTS_DIR', temp_dir / "results"):
        yield


class TestCheckpointManagerInit:
    """Test CheckpointManager initialization."""

    @pytest.mark.unit
    def test_initialization(self, temp_dir):
        """Test CheckpointManager initialization."""
        from src.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(task_name="test_task")

        assert manager.task_name == "test_task"
        assert manager.checkpoint_dir.exists()

    @pytest.mark.unit
    def test_creates_checkpoint_directory(self, temp_dir):
        """Test checkpoint directory is created on init."""
        from src.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(task_name="new_task")

        assert manager.checkpoint_dir.exists()
        assert "fine_tuned" in str(manager.checkpoint_dir)
        assert "new_task" in str(manager.checkpoint_dir)

    @pytest.mark.unit
    def test_loads_existing_registry(self, temp_dir):
        """Test loads existing registry from disk."""
        from src.utils.checkpoint_manager import CheckpointManager

        # Create a registry file first
        checkpoint_dir = temp_dir / "models" / "fine_tuned" / "existing_task"
        checkpoint_dir.mkdir(parents=True)
        registry_path = checkpoint_dir / "checkpoint_registry.json"
        registry_path.write_text(json.dumps({"existing_ckpt": {"path": "/test", "metrics": {}}}))

        manager = CheckpointManager(task_name="existing_task")

        assert "existing_ckpt" in manager.get_registry()


class TestCheckpointManagerRegister:
    """Test CheckpointManager register_checkpoint method."""

    @pytest.mark.unit
    def test_register_checkpoint(self, temp_dir, mock_checkpoint_metrics):
        """Test registering a checkpoint."""
        from src.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(task_name="test_task")
        ckpt_path = temp_dir / "checkpoint.pt"
        ckpt_path.touch()

        manager.register_checkpoint(
            checkpoint_path=ckpt_path,
            metrics=mock_checkpoint_metrics,
            epoch=5
        )

        assert "checkpoint.pt" in manager.list_checkpoints()

    @pytest.mark.unit
    def test_register_checkpoint_with_metrics(self, temp_dir, mock_checkpoint_metrics):
        """Test checkpoint metrics are stored correctly."""
        from src.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(task_name="test_task")
        ckpt_path = temp_dir / "checkpoint.pt"
        ckpt_path.touch()

        manager.register_checkpoint(
            checkpoint_path=ckpt_path,
            metrics=mock_checkpoint_metrics
        )

        entry = manager.get_checkpoint_entry("checkpoint.pt")
        assert entry["metrics"]["wer"] == 25.5
        assert entry["metrics"]["cer"] == 12.3
        assert entry["metrics"]["bleu"] == 65.0

    @pytest.mark.unit
    def test_register_checkpoint_with_epoch_and_step(self, temp_dir, mock_checkpoint_metrics):
        """Test checkpoint epoch and step are stored."""
        from src.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(task_name="test_task")
        ckpt_path = temp_dir / "checkpoint.pt"
        ckpt_path.touch()

        manager.register_checkpoint(
            checkpoint_path=ckpt_path,
            metrics=mock_checkpoint_metrics,
            epoch=10,
            step=5000
        )

        entry = manager.get_checkpoint_entry("checkpoint.pt")
        assert entry["epoch"] == 10
        assert entry["step"] == 5000

    @pytest.mark.unit
    def test_register_checkpoint_persists(self, temp_dir, mock_checkpoint_metrics):
        """Test registered checkpoint is persisted to disk."""
        from src.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(task_name="persist_test")
        ckpt_path = temp_dir / "checkpoint.pt"
        ckpt_path.touch()

        manager.register_checkpoint(
            checkpoint_path=ckpt_path,
            metrics=mock_checkpoint_metrics
        )

        # Create new manager instance
        manager2 = CheckpointManager(task_name="persist_test")
        assert "checkpoint.pt" in manager2.list_checkpoints()


class TestCheckpointManagerGetBest:
    """Test CheckpointManager get_best_checkpoint method."""

    @pytest.mark.unit
    def test_get_best_checkpoint_min_wer(self, temp_dir):
        """Test getting best checkpoint by minimum WER."""
        from src.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(task_name="best_test")

        # Register multiple checkpoints
        for i, wer in enumerate([30.0, 20.0, 25.0]):
            ckpt_path = temp_dir / f"ckpt_{i}.pt"
            ckpt_path.touch()
            manager.register_checkpoint(
                checkpoint_path=ckpt_path,
                metrics={"wer": wer}
            )

        best = manager.get_best_checkpoint(metric="wer", mode="min")

        assert best["metrics"]["wer"] == 20.0

    @pytest.mark.unit
    def test_get_best_checkpoint_max_bleu(self, temp_dir):
        """Test getting best checkpoint by maximum BLEU."""
        from src.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(task_name="best_test")

        for i, bleu in enumerate([60.0, 80.0, 70.0]):
            ckpt_path = temp_dir / f"ckpt_{i}.pt"
            ckpt_path.touch()
            manager.register_checkpoint(
                checkpoint_path=ckpt_path,
                metrics={"bleu": bleu}
            )

        best = manager.get_best_checkpoint(metric="bleu", mode="max")

        assert best["metrics"]["bleu"] == 80.0

    @pytest.mark.unit
    def test_get_best_checkpoint_returns_none_if_empty(self, temp_dir):
        """Test returns None when no checkpoints registered."""
        from src.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(task_name="empty_test")

        best = manager.get_best_checkpoint(metric="wer", mode="min")

        assert best is None

    @pytest.mark.unit
    def test_get_best_checkpoint_returns_none_if_metric_missing(self, temp_dir):
        """Test returns None when requested metric not in any checkpoint."""
        from src.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(task_name="missing_metric")
        ckpt_path = temp_dir / "ckpt.pt"
        ckpt_path.touch()
        manager.register_checkpoint(
            checkpoint_path=ckpt_path,
            metrics={"wer": 25.0}  # No BLEU
        )

        best = manager.get_best_checkpoint(metric="bleu", mode="max")

        assert best is None


class TestCheckpointManagerConvert:
    """Test CheckpointManager convert_checkpoint method."""

    @pytest.mark.unit
    def test_convert_to_safetensors(self, temp_dir):
        """Test converting checkpoint to safetensors format."""
        from src.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(task_name="convert_test")
        src_path = temp_dir / "source.pt"
        src_path.write_bytes(b"dummy checkpoint data")

        output = manager.convert_checkpoint(src_path, target_format="safetensors")

        assert output.suffix == ".safetensors"
        assert output.exists()

    @pytest.mark.unit
    def test_convert_to_pt(self, temp_dir):
        """Test converting checkpoint to PyTorch format."""
        from src.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(task_name="convert_test")
        src_path = temp_dir / "source.bin"
        src_path.write_bytes(b"dummy checkpoint data")

        output = manager.convert_checkpoint(src_path, target_format="pt")

        assert output.suffix == ".pt"

    @pytest.mark.unit
    def test_convert_unsupported_format_raises(self, temp_dir):
        """Test converting to unsupported format raises ValueError."""
        from src.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(task_name="convert_test")
        src_path = temp_dir / "source.pt"
        src_path.write_bytes(b"dummy")

        with pytest.raises(ValueError, match="Unsupported target format"):
            manager.convert_checkpoint(src_path, target_format="unknown")

    @pytest.mark.unit
    def test_convert_onnx_raises_not_implemented(self, temp_dir):
        """Test ONNX conversion raises NotImplementedError."""
        from src.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(task_name="convert_test")
        src_path = temp_dir / "source.pt"
        src_path.write_bytes(b"dummy")

        with pytest.raises(NotImplementedError):
            manager.convert_checkpoint(src_path, target_format="onnx")


class TestCheckpointManagerRemove:
    """Test CheckpointManager remove_checkpoint method."""

    @pytest.mark.unit
    def test_remove_checkpoint_from_registry(self, temp_dir, mock_checkpoint_metrics):
        """Test removing checkpoint from registry."""
        from src.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(task_name="remove_test")
        ckpt_path = temp_dir / "to_remove.pt"
        ckpt_path.touch()
        manager.register_checkpoint(
            checkpoint_path=ckpt_path,
            metrics=mock_checkpoint_metrics
        )

        manager.remove_checkpoint("to_remove.pt")

        assert "to_remove.pt" not in manager.list_checkpoints()

    @pytest.mark.unit
    def test_remove_checkpoint_deletes_file(self, temp_dir, mock_checkpoint_metrics):
        """Test removing checkpoint can delete file."""
        from src.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(task_name="remove_test")
        ckpt_path = temp_dir / "to_delete.pt"
        ckpt_path.touch()
        manager.register_checkpoint(
            checkpoint_path=ckpt_path,
            metrics=mock_checkpoint_metrics
        )

        manager.remove_checkpoint("to_delete.pt", delete_file=True)

        assert not ckpt_path.exists()


class TestCheckpointManagerUpdateMetrics:
    """Test CheckpointManager update_metrics method."""

    @pytest.mark.unit
    def test_update_metrics(self, temp_dir, mock_checkpoint_metrics):
        """Test updating checkpoint metrics."""
        from src.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(task_name="update_test")
        ckpt_path = temp_dir / "ckpt.pt"
        ckpt_path.touch()
        manager.register_checkpoint(
            checkpoint_path=ckpt_path,
            metrics=mock_checkpoint_metrics
        )

        manager.update_metrics("ckpt.pt", {"wer": 20.0, "new_metric": 99.0})

        entry = manager.get_checkpoint_entry("ckpt.pt")
        assert entry["metrics"]["wer"] == 20.0
        assert entry["metrics"]["new_metric"] == 99.0
        # Original metrics should be preserved/updated
        assert "cer" in entry["metrics"]


class TestCheckpointManagerList:
    """Test CheckpointManager list methods."""

    @pytest.mark.unit
    def test_list_checkpoints(self, temp_dir, mock_checkpoint_metrics):
        """Test listing all checkpoints."""
        from src.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(task_name="list_test")

        for name in ["ckpt_1.pt", "ckpt_2.pt", "ckpt_3.pt"]:
            ckpt_path = temp_dir / name
            ckpt_path.touch()
            manager.register_checkpoint(
                checkpoint_path=ckpt_path,
                metrics=mock_checkpoint_metrics
            )

        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) == 3
        assert "ckpt_1.pt" in checkpoints
        assert "ckpt_2.pt" in checkpoints
        assert "ckpt_3.pt" in checkpoints

    @pytest.mark.unit
    def test_get_registry(self, temp_dir, mock_checkpoint_metrics):
        """Test getting full registry."""
        from src.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(task_name="registry_test")
        ckpt_path = temp_dir / "ckpt.pt"
        ckpt_path.touch()
        manager.register_checkpoint(
            checkpoint_path=ckpt_path,
            metrics=mock_checkpoint_metrics
        )

        registry = manager.get_registry()

        assert isinstance(registry, dict)
        assert "ckpt.pt" in registry
