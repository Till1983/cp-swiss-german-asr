"""Seq2Seq trainer with an Elastic Weight Consolidation (EWC) penalty.

This reparents the capstone's ``EWCTrainer`` pattern (originally a subclass of
the plain ``Trainer`` in ``scripts/train_german_adaptation.py``) onto
``Seq2SeqTrainer`` so it can be used for Whisper generation-based evaluation.

Correctness requirements (designed in here, not bolted on later):

* **Requirement A -- the 1/2 factor.** ``fisher_diagonal.pt`` contains the
  ``1/N`` average of squared gradients with *no* ``1/2`` applied (see
  ``scripts/compute_fisher.py``'s docstring -- that is the authoritative
  source). The ``1/2`` is this training loop's responsibility. We apply it
  explicitly (``apply_half_factor=True``):

      ewc_loss = 0.5 * lambda * sum_i F_i * (theta_i - theta_i*)^2

  The *raw* EWC term logged for lambda calibration is the un-halved, unscaled
  sum -- the 1/2 and lambda are applied only when adding to the task loss.

* **Requirement B -- fp32 accumulation.** ``fisher_dict`` and ``old_params``
  are fp32 while the model trains in bf16. The per-parameter difference,
  square, Fisher weighting, and the sum over ~1.5B terms are all done in fp32
  and accumulated into an fp32 scalar. Summing that many terms in bf16
  (~3 significant digits) would silently round the penalty to zero without
  raising an error.

* **Requirement C -- key coverage.** At ``__init__`` we assert every model
  parameter name is present in both ``fisher_dict`` and ``old_params``. A
  silent partial match means EWC penalises fewer parameters than intended; an
  empty match means it does nothing -- neither raises on its own, so we raise
  here.
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import Seq2SeqTrainer

logger = logging.getLogger(__name__)


def assert_key_coverage(param_names, fisher_dict, old_params, strict: bool = True):
    """Requirement C: every model parameter must be covered by Fisher + theta*.

    Returns ``None`` on success; raises ``ValueError`` (or warns, if not
    ``strict``) when any model parameter name is missing from either dict.
    """
    param_names = set(param_names)
    missing_from_fisher = sorted(param_names - set(fisher_dict.keys()))
    missing_from_theta = sorted(param_names - set(old_params.keys()))

    if missing_from_fisher or missing_from_theta:
        msg = (
            "EWC key-coverage check failed -- the penalty would silently "
            "cover fewer parameters than the model has.\n"
            f"  model parameters: {len(param_names)}\n"
            f"  fisher_dict keys: {len(fisher_dict)} "
            f"(missing {len(missing_from_fisher)})\n"
            f"  old_params keys:  {len(old_params)} "
            f"(missing {len(missing_from_theta)})\n"
        )
        if missing_from_fisher:
            msg += f"  first missing from fisher: {missing_from_fisher[:5]}\n"
        if missing_from_theta:
            msg += f"  first missing from theta*: {missing_from_theta[:5]}\n"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
        return msg
    logger.info(
        "EWC key-coverage check passed: all %d model parameters are covered "
        "by Fisher and theta*.",
        len(param_names),
    )
    return None


def ewc_raw_term(named_parameters, fisher_dict, old_params) -> torch.Tensor:
    """Un-halved, unscaled fp32 sum_i F_i * (theta_i - theta_i*)^2.

    Requirement B: the difference, square, Fisher weighting, and the sum are
    all done in fp32 and accumulated into an fp32 scalar, even when the live
    parameters are bf16. Differentiable w.r.t. the live parameters.
    """
    total = None
    for name, param in named_parameters:
        if name not in fisher_dict:
            continue
        fisher = fisher_dict[name].to(param.device).float()
        theta_star = old_params[name].to(param.device).float()
        diff = param.float() - theta_star  # fp32, keeps grad to param
        contrib = (fisher * diff.pow(2)).sum()
        total = contrib if total is None else total + contrib
    if total is None:
        # No covered parameters -> zero (on a best-effort device).
        device = "cpu"
        return torch.zeros((), dtype=torch.float32, device=device)
    return total


def scaled_ewc_penalty(
    raw_term: torch.Tensor, ewc_lambda: float, apply_half_factor: bool
) -> torch.Tensor:
    """Apply lambda and (optionally) the 1/2 factor to the raw EWC term.

    Requirement A: the raw term itself is never halved/scaled; the 1/2 and
    lambda are applied only here, when forming the penalty added to task loss.
    """
    factor = 0.5 if apply_half_factor else 1.0
    return factor * ewc_lambda * raw_term


def load_fisher_and_theta(
    fisher_path,
    theta_star_path,
    map_location: str = "cpu",
):
    """Load ``fisher_diagonal.pt`` / ``theta_star.pt`` from disk as fp32 dicts.

    These tensors are saved fp32 by ``compute_fisher.py`` regardless of the
    fine-tuning precision; we cast defensively to fp32 anyway so the EWC
    accumulation is fp32 even if a future Fisher run changes dtype.
    """
    fisher_dict = torch.load(fisher_path, map_location=map_location)
    old_params = torch.load(theta_star_path, map_location=map_location)
    fisher_dict = {k: v.detach().float() for k, v in fisher_dict.items()}
    old_params = {k: v.detach().float() for k, v in old_params.items()}
    return fisher_dict, old_params


class Seq2SeqEWCTrainer(Seq2SeqTrainer):
    """``Seq2SeqTrainer`` with a diagonal-Fisher EWC penalty on the loss.

    Args:
        fisher_dict: ``{param_name: fp32 Fisher diagonal tensor}``.
        old_params: ``{param_name: fp32 reference (theta*) tensor}``.
        ewc_lambda: Penalty weight (lambda). 1.0 is a fine placeholder for the
            smoke test; the real value comes from the calibration log later.
        apply_half_factor: If True (Requirement A), multiply the penalty by 1/2
            in addition to lambda. The raw logged term is always un-halved.
        ewc_log_path: CSV file to append per-step unscaled ``task_loss`` and
            raw (un-halved, unscaled) ``ewc_term`` for lambda calibration.
        strict_key_coverage: If True, raise when any model parameter is missing
            from ``fisher_dict``/``old_params`` (Requirement C).
    """

    def __init__(
        self,
        *args,
        fisher_dict: Dict[str, torch.Tensor],
        old_params: Dict[str, torch.Tensor],
        ewc_lambda: float = 1.0,
        apply_half_factor: bool = True,
        ewc_log_path: Optional[Any] = None,
        strict_key_coverage: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.ewc_lambda = float(ewc_lambda)
        self.apply_half_factor = bool(apply_half_factor)
        self.ewc_log_path = Path(ewc_log_path) if ewc_log_path is not None else None

        # Requirement B: hold Fisher / theta* in fp32.
        self.fisher_dict = {k: v.detach().float() for k, v in fisher_dict.items()}
        self.old_params = {k: v.detach().float() for k, v in old_params.items()}

        # Requirement C: runtime key-coverage guardrail.
        self._validate_key_coverage(strict=strict_key_coverage)

        self._ewc_on_device = False
        self._ewc_log_header_written = False

    # ------------------------------------------------------------------
    # Requirement C
    # ------------------------------------------------------------------
    def _validate_key_coverage(self, strict: bool = True) -> None:
        param_names = [name for name, _ in self.model.named_parameters()]
        assert_key_coverage(param_names, self.fisher_dict, self.old_params, strict=strict)

    # ------------------------------------------------------------------
    # Requirement B: move Fisher / theta* to the model device once.
    # ------------------------------------------------------------------
    def _ensure_ewc_on_device(self, device: torch.device) -> None:
        if self._ewc_on_device:
            return
        self.fisher_dict = {k: v.to(device) for k, v in self.fisher_dict.items()}
        self.old_params = {k: v.to(device) for k, v in self.old_params.items()}
        self._ewc_on_device = True

    def compute_raw_ewc_term(self, model) -> torch.Tensor:
        """Un-halved, unscaled fp32 EWC term (delegates to :func:`ewc_raw_term`)."""
        device = next(model.parameters()).device
        self._ensure_ewc_on_device(device)
        return ewc_raw_term(model.named_parameters(), self.fisher_dict, self.old_params)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if num_items_in_batch is not None:
            task_loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
        else:
            task_loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True
            )

        # Only apply / log the penalty during training (skip eval forwards).
        if model.training and self.fisher_dict:
            raw_ewc = self.compute_raw_ewc_term(model)
            ewc_penalty = scaled_ewc_penalty(
                raw_ewc, self.ewc_lambda, self.apply_half_factor
            )
            total_loss = task_loss + ewc_penalty.to(task_loss.dtype)
            self._log_calibration(task_loss.detach(), raw_ewc.detach())
        else:
            total_loss = task_loss

        return (total_loss, outputs) if return_outputs else total_loss

    # ------------------------------------------------------------------
    # Calibration logging: unscaled task_loss + raw (un-halved) EWC term.
    # ------------------------------------------------------------------
    def _log_calibration(
        self, task_loss: torch.Tensor, raw_ewc_term: torch.Tensor
    ) -> None:
        if self.ewc_log_path is None:
            return
        # Only the main process writes, to avoid interleaved rows under DDP.
        try:
            if not self.is_world_process_zero():
                return
        except Exception:
            pass

        row = {
            "global_step": int(self.state.global_step),
            "task_loss": float(task_loss.item()),
            "ewc_term_raw": float(raw_ewc_term.item()),
            "ewc_lambda": self.ewc_lambda,
            "apply_half_factor": self.apply_half_factor,
        }

        self.ewc_log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.ewc_log_path.exists()
        # Append incrementally so a mid-run crash still leaves a usable log.
        with open(self.ewc_log_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header and not self._ewc_log_header_written:
                writer.writeheader()
                self._ewc_log_header_written = True
            writer.writerow(row)
