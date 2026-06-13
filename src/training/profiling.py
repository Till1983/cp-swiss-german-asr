"""Reusable GPU/VRAM profiling instrumentation for training runs.

These helpers are built to be reused by the real baseline and EWC-grid runs,
not thrown away after the smoke test. Every CUDA-touching path guards on
``torch.cuda.is_available()`` so the same code is a no-op on a CPU-only dev
machine (and is unit-tested in both branches).
"""

import csv
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class MemoryProfilerCallback(TrainerCallback):
    """Record peak CUDA memory per step, tagged by train/eval phase.

    On each step end (and each eval prediction step, when the installed
    transformers exposes ``on_prediction_step``) we record
    ``torch.cuda.max_memory_allocated`` / ``max_memory_reserved`` and then
    reset the peak stats, so each row is the peak *since the previous record*.

    Rows are appended incrementally (the file is reopened per write) so a
    mid-run OOM still leaves a usable partial log rather than losing an
    end-of-run buffer.

    On a machine without CUDA the callback no-ops gracefully.
    """

    FIELDNAMES = [
        "global_step",
        "phase",
        "max_allocated_bytes",
        "max_reserved_bytes",
        "max_allocated_gb",
        "max_reserved_gb",
    ]

    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.enabled = torch.cuda.is_available()
        self._header_written = False

    def _record(self, phase: str, state) -> None:
        if not self.enabled:
            return

        max_allocated = torch.cuda.max_memory_allocated()
        max_reserved = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()

        row = {
            "global_step": int(getattr(state, "global_step", 0) or 0),
            "phase": phase,
            "max_allocated_bytes": int(max_allocated),
            "max_reserved_bytes": int(max_reserved),
            "max_allocated_gb": round(max_allocated / (1024**3), 4),
            "max_reserved_gb": round(max_reserved / (1024**3), 4),
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.output_path.exists()
        with open(self.output_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.FIELDNAMES)
            if write_header and not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)

    def on_step_end(self, args, state, control, **kwargs):
        self._record("train", state)

    def on_prediction_step(self, args, state, control, **kwargs):
        self._record("eval", state)


class NvidiaSmiLogger:
    """Launch ``nvidia-smi`` polling as a background subprocess logging to CSV.

    Wraps ``nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
    -l <interval>`` with a clean start/stop lifecycle. Usable as a context
    manager. Gracefully no-ops if ``nvidia-smi`` is not on PATH.
    """

    def __init__(self, output_path, interval: int = 1):
        self.output_path = Path(output_path)
        self.interval = int(interval)
        self._proc: Optional[subprocess.Popen] = None
        self._fh = None

    def start(self) -> bool:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "nvidia-smi",
            "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu",
            "--format=csv,nounits",
            "-l",
            str(self.interval),
        ]
        try:
            self._fh = open(self.output_path, "w")
            self._proc = subprocess.Popen(cmd, stdout=self._fh)
        except (FileNotFoundError, OSError) as exc:
            logger.warning("nvidia-smi logging unavailable: %s", exc)
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            self._proc = None
            return False
        return True

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False


class CudaMemoryHistoryRecorder:
    """Record a CUDA allocator history snapshot for the first N steps.

    Wraps ``torch.cuda.memory._record_memory_history(enabled="all")`` and
    ``torch.cuda.memory._dump_snapshot(path)``. The snapshot can be loaded in
    PyTorch's memory visualizer. No-ops without CUDA.
    """

    def __init__(self, output_path, max_entries: int = 100_000):
        self.output_path = Path(output_path)
        self.max_entries = int(max_entries)
        self.enabled = torch.cuda.is_available()
        self._recording = False

    def start(self) -> bool:
        if not self.enabled:
            return False
        torch.cuda.memory._record_memory_history(
            enabled="all", max_entries=self.max_entries
        )
        self._recording = True
        return True

    def dump(self) -> Optional[Path]:
        if not self.enabled or not self._recording:
            return None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(self.output_path))
        return self.output_path

    def stop(self) -> None:
        if not self.enabled or not self._recording:
            return
        # Passing enabled=None disables history recording.
        torch.cuda.memory._record_memory_history(enabled=None)
        self._recording = False


class ThroughputTracker:
    """Track wall-clock seconds/step and extrapolate to a target step count."""

    def __init__(self):
        self._start: Optional[float] = None
        self._first_step: Optional[int] = None
        self._last_step: int = 0

    def on_first_step(self, global_step: int) -> None:
        self._start = time.time()
        self._first_step = global_step

    def update(self, global_step: int) -> None:
        self._last_step = global_step

    def summary(self, target_steps: Optional[int] = None) -> dict:
        if self._start is None or self._first_step is None:
            return {}
        elapsed = time.time() - self._start
        steps = max(1, self._last_step - self._first_step)
        sec_per_step = elapsed / steps
        out = {
            "elapsed_seconds": elapsed,
            "steps_measured": steps,
            "sec_per_step": sec_per_step,
        }
        if target_steps is not None:
            out["extrapolated_seconds_for_target"] = sec_per_step * target_steps
            out["target_steps"] = target_steps
        return out
