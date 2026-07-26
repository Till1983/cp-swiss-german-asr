# Whisper Large-v2 + EWC grid run (lambda=300000.0) — summary

- attention: sdpa (FA2 unavailable on Blackwell)
- gradient_checkpointing: False
- per_device_train_batch_size: 16
- max_steps: None
- ewc_lambda: 300000.0
- EWC half-factor applied: True (Requirement A: fisher_diagonal.pt has NO 1/2 baked in)

## VRAM
- peak allocated: 87.57 GB
- peak reserved: 90.51 GB
- budget: 96 GB (see vram_profile.csv for per-step train/eval peaks)

## Outcome
- training completed without OOM.
- final metrics: {'train_runtime': 2041.3791, 'train_samples_per_second': 9.859, 'train_steps_per_second': 0.617, 'total_flos': 4.27288167936e+19, 'train_loss': 0.46743660654340474, 'epoch': 5.0}
- go/no-go: inspect loss_curve.csv for a sane decrease through warmup (no spike/flatline) and ewc_calibration.csv for the raw EWC term scale.
