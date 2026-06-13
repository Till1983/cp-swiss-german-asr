"""Unit tests for Seq2SeqEWCTrainer and its EWC helper functions.

All synthetic: a tiny WhisperConfig is instantiated on CPU so the EWC
correctness requirements (A: the 1/2 factor, B: fp32 accumulation under a bf16
model, C: key coverage, and the raw-logging path) are pinned down in seconds
without GPU or the real Fisher tensors.
"""

import csv

import pytest
import torch
from transformers import (
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperForConditionalGeneration,
)

from src.training.ewc_trainer import (
    Seq2SeqEWCTrainer,
    assert_key_coverage,
    ewc_raw_term,
    scaled_ewc_penalty,
)


def tiny_config():
    return WhisperConfig(
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        num_mel_bins=80,
        max_source_positions=50,
        max_target_positions=50,
        vocab_size=120,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        decoder_start_token_id=1,
    )


@pytest.fixture
def tiny_model():
    return WhisperForConditionalGeneration(tiny_config())


# ---------------------------------------------------------------------------
# Requirement A: the 1/2 factor + lambda scaling (raw term left untouched)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_half_factor_applied_exactly():
    raw = torch.tensor(4.0)
    # With lambda=2: half-factor -> 0.5 * 2 * 4 = 4.0 ; no half -> 2 * 4 = 8.0
    assert scaled_ewc_penalty(raw, ewc_lambda=2.0, apply_half_factor=True).item() == pytest.approx(4.0)
    assert scaled_ewc_penalty(raw, ewc_lambda=2.0, apply_half_factor=False).item() == pytest.approx(8.0)


@pytest.mark.unit
def test_penalty_linear_in_lambda():
    raw = torch.tensor(3.0)
    p1 = scaled_ewc_penalty(raw, ewc_lambda=1.0, apply_half_factor=True).item()
    p10 = scaled_ewc_penalty(raw, ewc_lambda=10.0, apply_half_factor=True).item()
    assert p10 == pytest.approx(10.0 * p1)


# ---------------------------------------------------------------------------
# Requirement B: fp32 accumulation even with a bf16 model
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_fp32_accumulation_with_bf16_model():
    model = WhisperForConditionalGeneration(tiny_config()).to(torch.bfloat16)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.5)  # 1.5 is exactly representable in bf16

    fisher = {n: torch.full_like(p, 2.0, dtype=torch.float32)
              for n, p in model.named_parameters()}
    old_params = {n: torch.zeros_like(p, dtype=torch.float32)
                  for n, p in model.named_parameters()}

    raw = ewc_raw_term(model.named_parameters(), fisher, old_params)

    # per element: fisher * (1.5 - 0)^2 = 2.0 * 2.25 = 4.5
    numel = sum(p.numel() for _, p in model.named_parameters())
    expected = 4.5 * numel

    assert raw.dtype == torch.float32
    assert raw.item() > 0  # would be silently ~0 if accumulated in bf16 wrongly
    assert raw.item() == pytest.approx(expected, rel=1e-4)


@pytest.mark.unit
def test_ewc_term_is_differentiable_wrt_params():
    model = WhisperForConditionalGeneration(tiny_config())
    fisher = {n: torch.ones_like(p, dtype=torch.float32)
              for n, p in model.named_parameters()}
    old_params = {n: torch.zeros_like(p, dtype=torch.float32)
                  for n, p in model.named_parameters()}

    raw = ewc_raw_term(model.named_parameters(), fisher, old_params)
    raw.backward()
    # Gradient must flow back to the live parameters.
    grads = [p.grad for _, p in model.named_parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum().item() > 0 for g in grads)


# ---------------------------------------------------------------------------
# Requirement C: key-coverage assertion
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_key_coverage_passes_when_complete(tiny_model):
    names = [n for n, _ in tiny_model.named_parameters()]
    fisher = {n: torch.ones(1) for n in names}
    old = {n: torch.ones(1) for n in names}
    # Should not raise.
    assert assert_key_coverage(names, fisher, old, strict=True) is None


@pytest.mark.unit
def test_key_coverage_fires_when_key_missing(tiny_model):
    names = [n for n, _ in tiny_model.named_parameters()]
    fisher = {n: torch.ones(1) for n in names}
    old = {n: torch.ones(1) for n in names}
    removed = names[0]
    del fisher[removed]
    with pytest.raises(ValueError, match="key-coverage"):
        assert_key_coverage(names, fisher, old, strict=True)


# ---------------------------------------------------------------------------
# Trainer-level: key coverage at __init__ and raw calibration logging
# ---------------------------------------------------------------------------
def _make_trainer(tmp_path, model, should_drop_key=False, ewc_lambda=1.0, log_path=None):
    from transformers import Seq2SeqTrainingArguments

    names = [n for n, _ in model.named_parameters()]
    fisher = {n: torch.ones_like(p, dtype=torch.float32)
              for n, p in model.named_parameters()}
    old = {n: torch.zeros_like(p, dtype=torch.float32)
           for n, p in model.named_parameters()}
    if should_drop_key:
        del fisher[names[0]]

    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path / "out"),
        report_to=[],
        use_cpu=True,
        per_device_train_batch_size=1,
        logging_steps=1,
    )
    return Seq2SeqEWCTrainer(
        model=model,
        args=args,
        fisher_dict=fisher,
        old_params=old,
        ewc_lambda=ewc_lambda,
        ewc_log_path=log_path,
    )


@pytest.mark.unit
def test_trainer_init_raises_on_missing_key(tmp_path, tiny_model):
    with pytest.raises(ValueError, match="key-coverage"):
        _make_trainer(tmp_path, tiny_model, drop_key=True)


@pytest.mark.unit
def test_trainer_init_passes_with_full_coverage(tmp_path, tiny_model):
    trainer = _make_trainer(tmp_path, tiny_model)
    assert trainer.ewc_lambda == 1.0
    assert trainer.apply_half_factor is True


@pytest.mark.unit
def test_calibration_log_records_unscaled_values(tmp_path, tiny_model):
    log_path = tmp_path / "ewc_calibration.csv"
    # lambda far from 1 so any accidental scaling of the logged values shows up.
    trainer = _make_trainer(tmp_path, tiny_model, ewc_lambda=100.0, log_path=log_path)
    trainer._log_calibration(torch.tensor(2.0), torch.tensor(9.0))

    assert log_path.exists()
    with open(log_path) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert float(rows[0]["task_loss"]) == pytest.approx(2.0)
    # Raw EWC term is logged un-halved and unscaled (NOT multiplied by lambda).
    assert float(rows[0]["ewc_term_raw"]) == pytest.approx(9.0)
    assert float(rows[0]["ewc_lambda"]) == pytest.approx(100.0)
