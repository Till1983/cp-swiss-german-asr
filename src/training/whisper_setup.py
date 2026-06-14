"""Model/config setup helpers for the Whisper Swiss German trainer.

These live in ``src/`` (not in the ``scripts/`` CLI) so they are importable by
the unit tests -- the ``test-unit`` Docker service mounts ``src`` and ``tests``
but not ``scripts``.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_path(base_dir, relative) -> Path:
    """Join a config-relative path onto one of the ``config.*`` base dirs.

    Per the project's path-resolution convention, every ``*_path``/``*_dir``/
    ``*_metadata`` value in the training config is relative to ``DATA_DIR``,
    ``MODELS_DIR`` or ``RESULTS_DIR``. An already-absolute path is returned
    unchanged (lets a CLI flag pass an explicit absolute override).
    """
    relative = Path(relative)
    if relative.is_absolute():
        return relative
    return Path(base_dir) / relative


def configure_caching(model, gradient_checkpointing: bool) -> None:
    """Set ``use_cache`` correctly for training vs. generation (Requirement D).

    Gradient checkpointing is incompatible with the decoder KV cache, so when
    GC is on we must set ``config.use_cache=False`` for training. Generation,
    however, needs the cache to be efficient, so we keep
    ``generation_config.use_cache=True`` regardless -- ``generate()`` reads the
    generation config, so it re-enables the cache for eval even when training
    ran with it off.
    """
    model.config.use_cache = not gradient_checkpointing
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True


def set_decoding_config(
    model,
    processor,
    language: str = "de",
    task: str = "transcribe",
    condition_on_prev_tokens: bool = False,
) -> None:
    """Force language/task decoding and disable prev-token conditioning.

    ``condition_on_prev_tokens=False`` guards against the hallucination loop
    seen in the capstone (469% WER on an AG sample without it).

    NOTE on ``forced_decoder_ids`` vs ``language``/``task``: recent
    ``transformers`` versions treat having both ``generation_config.language``/
    ``.task`` AND a non-None ``generation_config.forced_decoder_ids`` set as
    conflicting at ``generate()`` time (warning or error depending on version).
    ``generate()`` reads ``generation_config``, so that's the authoritative
    path -- we set language/task there and explicitly null out
    ``forced_decoder_ids`` on the generation config. ``model.config`` keeps
    ``forced_decoder_ids`` for any legacy code path that still reads it from
    there; this does not affect ``generate()``.
    """
    processor.tokenizer.set_prefix_tokens(language=language, task=task)

    forced_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    model.config.forced_decoder_ids = forced_ids
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.language = language
        model.generation_config.task = task
        # Avoid the both-set conflict described above; language/task on
        # generation_config is what actually drives generate().
        model.generation_config.forced_decoder_ids = None

    # Disable prev-token conditioning on both config and generation_config.
    for cfg in (model.config, getattr(model, "generation_config", None)):
        if cfg is not None:
            setattr(cfg, "condition_on_prev_tokens", condition_on_prev_tokens)


def build_whisper_model(
    model_cfg: Dict[str, Any],
    augmentation_cfg: Optional[Dict[str, Any]] = None,
    gradient_checkpointing: bool = False,
):
    """Instantiate the Whisper model + processor from a config dict.

    Loads with the configured attention implementation (``sdpa`` -- FA2 is
    unavailable on the Blackwell setup) and dtype. For training runs,
    ``model_cfg["torch_dtype"]`` should be ``"float32"`` (fp32 master weights;
    ``training.bf16=True`` in the trainer args then gives bf16 autocast
    compute on top -- this is the standard Whisper fine-tuning recipe and is
    what the VRAM budget in the smoke test assumes). ``"bfloat16"`` here would
    remove the fp32 master copy and roughly halve the static VRAM floor,
    invalidating the smoke test's headroom measurement. The zero-shot
    inference path may still load bf16 directly; that's a separate config.

    Also forces de/transcribe decoding, optionally enables SpecAugment, and
    sets ``use_cache`` per the GC flag.

    Args:
        model_cfg: the ``cfg["model"]`` sub-dict.
        augmentation_cfg: the ``cfg["augmentation"]`` sub-dict (or ``None``).
            Read separately from ``model_cfg`` because the YAML keeps
            augmentation settings under their own top-level section --
            ``model_cfg`` alone never contains ``spec_augment``.
        gradient_checkpointing: enable activation checkpointing.
    """
    name = model_cfg["name"]
    attn = model_cfg.get("attn_implementation", "sdpa")
    dtype = _DTYPE_MAP.get(str(model_cfg.get("torch_dtype", "float32")).lower())

    processor = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(
        name,
        attn_implementation=attn,
        torch_dtype=dtype,
    )

    if (augmentation_cfg or {}).get("spec_augment"):
        model.config.apply_spec_augment = True
        # LibriSpeech Basic (Park et al. 2019, Table 1, §2: W=80 F=27 mF=1 T=100 p=1.0 mT=1),
        # translated to HF's wav2vec2-style probabilistic masking by matching LB's
        # EXPECTED masked extent on Whisper's 80x3000 input-feature grid:
        #   time: E[t]=T/2=50 frames / 3000  -> length 50, prob 50/3000  ~= 0.017, ~1 mask
        #   freq: E[f]=F/2=13.5 ch  / 80      -> length 14, prob 13.5/80 ~= 0.17,  ~1 mask
        # Mask GEOMETRY differs from LB (fixed-width, prob-placed masks vs LB's single
        # uniform-width mask); expected COVERAGE matches. Time warp (W=80) dropped:
        # HF does not implement it; Park et al. §5 recommend it as first to drop.
        model.config.mask_time_prob = 0.017
        model.config.mask_time_length = 50
        model.config.mask_time_min_masks = 1  # override HF Whisper default of 2 (LB mT=1)
        model.config.mask_feature_prob = 0.17
        model.config.mask_feature_length = 14
        model.config.mask_feature_min_masks = 0

    set_decoding_config(
        model,
        processor,
        language=model_cfg.get("language", "de"),
        task=model_cfg.get("task", "transcribe"),
        condition_on_prev_tokens=model_cfg.get("condition_on_prev_tokens", False),
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    configure_caching(model, gradient_checkpointing)

    return model, processor