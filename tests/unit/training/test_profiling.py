"""Unit tests for profiling instrumentation (CUDA-guarded; no GPU required)."""

import csv
from types import SimpleNamespace

import pytest
import torch

from src.training.profiling import (
    CudaMemoryHistoryRecorder,
    MemoryProfilerCallback,
    NvidiaSmiLogger,
    ThroughputTracker,
)


# ---------------------------------------------------------------------------
# MemoryProfilerCallback — disabled (no CUDA) and enabled (mocked CUDA) branches
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_memory_callback_noops_without_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cb = MemoryProfilerCallback(tmp_path / "vram.csv")
    assert cb.enabled is False
    cb.on_step_end(None, SimpleNamespace(global_step=1), None)
    # Nothing should be written on a CPU-only machine.
    assert not (tmp_path / "vram.csv").exists()


@pytest.mark.unit
def test_memory_callback_records_train_and_eval_phases(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 2 * 1024**3)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a, **k: 3 * 1024**3)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)

    cb = MemoryProfilerCallback(tmp_path / "vram.csv")
    assert cb.enabled is True
    cb.on_step_end(None, SimpleNamespace(global_step=5), None)
    cb.on_prediction_step(None, SimpleNamespace(global_step=5), None)

    with open(tmp_path / "vram.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["phase"] for r in rows] == ["train", "eval"]
    assert float(rows[0]["max_allocated_gb"]) == pytest.approx(2.0)
    assert float(rows[0]["max_reserved_gb"]) == pytest.approx(3.0)


@pytest.mark.unit
def test_memory_callback_appends_incrementally(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 1024**3)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a, **k: 1024**3)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)

    cb = MemoryProfilerCallback(tmp_path / "vram.csv")
    cb.on_step_end(None, SimpleNamespace(global_step=1), None)
    cb.on_step_end(None, SimpleNamespace(global_step=2), None)
    with open(tmp_path / "vram.csv") as fh:
        rows = list(csv.DictReader(fh))
    # Two rows, single header (incremental append leaves a usable partial log).
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# NvidiaSmiLogger — start/stop lifecycle with a mocked subprocess
# ---------------------------------------------------------------------------
class _FakeProc:
    def __init__(self):
        self.terminated = False
        self.waited = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True

    def kill(self):
        pass


@pytest.mark.unit
def test_nvidia_smi_logger_lifecycle(tmp_path, monkeypatch):
    import src.training.profiling as profiling

    fake = _FakeProc()
    calls = {}

    def fake_popen(cmd, stdout=None):
        calls["cmd"] = cmd
        return fake

    monkeypatch.setattr(profiling.subprocess, "Popen", fake_popen)

    logger = NvidiaSmiLogger(tmp_path / "smi.csv", interval=1)
    assert logger.start() is True
    assert "nvidia-smi" in calls["cmd"][0]
    logger.stop()
    assert fake.terminated is True
    assert fake.waited is True


@pytest.mark.unit
def test_nvidia_smi_logger_handles_missing_binary(tmp_path, monkeypatch):
    import src.training.profiling as profiling

    def boom(cmd, stdout=None):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(profiling.subprocess, "Popen", boom)
    logger = NvidiaSmiLogger(tmp_path / "smi.csv")
    assert logger.start() is False
    logger.stop()  # must not raise


# ---------------------------------------------------------------------------
# CudaMemoryHistoryRecorder — no-CUDA guard
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_memory_history_recorder_noops_without_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rec = CudaMemoryHistoryRecorder(tmp_path / "snap.pickle")
    assert rec.enabled is False
    assert rec.start() is False
    assert rec.dump() is None
    rec.stop()  # must not raise
    assert not (tmp_path / "snap.pickle").exists()


# ---------------------------------------------------------------------------
# ThroughputTracker
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_throughput_tracker_extrapolates():
    t = ThroughputTracker()
    t.on_first_step(0)
    t.update(10)
    summary = t.summary(target_steps=100)
    assert summary["steps_measured"] == 10
    assert "sec_per_step" in summary
    assert summary["target_steps"] == 100
    assert summary["extrapolated_seconds_for_target"] == pytest.approx(
        summary["sec_per_step"] * 100
    )
