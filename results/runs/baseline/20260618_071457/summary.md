# Whisper Large-v2 + non-EWC fine-tuning — summary

- attention: sdpa (FA2 unavailable on Blackwell)
- gradient_checkpointing: False
- per_device_train_batch_size: 16
- max_steps: None
- ewc_lambda (placeholder): 0.0
- EWC half-factor applied: True (Requirement A: fisher_diagonal.pt has NO 1/2 baked in)

## VRAM
- peak allocated: 69.29 GB
- peak reserved: 71.80 GB
- budget: 96 GB (see vram_profile.csv for per-step train/eval peaks)

## Outcome
- training completed without OOM.
- final metrics: {'train_runtime': 1872.9629, 'train_samples_per_second': 10.745, 'train_steps_per_second': 0.673, 'total_flos': 4.27288167936e+19, 'train_loss': 0.19371600309534678, 'epoch': 5.0}
- go/no-go on LR=1e-5/warmup=50: inspect loss_curve.csv for a sane decrease through warmup (no spike/flatline) and ewc_calibration.csv for the raw EWC term scale used to centre the lambda grid.
