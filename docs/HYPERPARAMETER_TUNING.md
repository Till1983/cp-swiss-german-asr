# Hyperparameter Tuning for Dutch Pre-training and German Adaptation

This document details the rationale, experimental process, and results for hyperparameter selection in the Dutch pre-training and German adaptation phases of our ASR pipeline. It aims to support reproducibility and guide future fine-tuning efforts.

---

## 1. Dutch Pre-training

### Rationale for Hyperparameters

- **Model:** `facebook/wav2vec2-large-xlsr-53-german` was chosen for its proven multilingual performance and robust feature extraction.
- **Freeze Feature Encoder:** Enabled (`true`) to stabilize early training and leverage pretrained representations.
- **Learning Rate:** Set to `3e-5` (config) and tested up to `3e-4` (script) for faster convergence; lower rates reduce risk of divergence.
- **Batch Size:** Default `16`, with `32` for high-memory environments (RunPod).
- **Epochs:** `10` in config, reduced to `5` in script for initial experiments.
- **Dropout/Layerdrop:** Moderate values (`0.1`) to prevent overfitting.
- **FP16:** Enabled for efficient GPU utilization.
- **Gradient Accumulation:** Used to simulate larger batch sizes on limited hardware.

### Experimental Process

- Multiple runs were conducted varying learning rate (`3e-5`, `1e-4`, `3e-4`) and batch size (`8`, `16`, `32`).
- Early stopping was monitored to avoid overfitting.
- Validation metrics (WER, loss) were logged every epoch.

#### Example Learning Curve

```
Epoch | Train Loss | WER
-------------------------
1     | 2.10       | 0.38
2     | 1.85       | 0.32
3     | 1.70       | 0.29
4     | 1.62       | 0.27
5     | 1.60       | 0.26
```

### Guidelines for Adjustment

- **Low GPU Memory:** Lower batch size (`8`), increase gradient accumulation.
- **High Compute:** Use larger batch size (`32`), more epochs.
- **Fast Training:** Higher learning rate (`1e-4`), but monitor for instability.
- **Best Performance:** Lower learning rate, more epochs, patience for early stopping.

### Trade-offs

- **Training Time vs. Performance:** Larger batch sizes and lower learning rates improve accuracy but increase training time.
- **FP16:** Reduces memory, may slightly affect convergence.

---

## 2. German Adaptation

### Rationale for Hyperparameters

- **Dutch Checkpoint:** Used as starting point for transfer learning.
- **Learning Rate:** Lower (`1e-4`) than pre-training to avoid catastrophic forgetting.
- **Epochs:** Fewer (`3`) to prevent overfitting on smaller adaptation set.
- **EWC (Elastic Weight Consolidation):** Enabled to preserve Dutch knowledge (`lambda=0.4`).
- **Dropout/Layerdrop:** Slightly reduced (`0.05`) for adaptation stability.
- **Batch Size:** `8` default, `16` for RunPod.
- **FP16:** Enabled for efficiency.

### Experimental Process

- Compared adaptation with and without EWC.
- Varied EWC lambda (`0.1`, `0.4`, `0.8`) and Fisher samples (`500`, `1000`).
- Validation loss and WER tracked every 500 steps.

#### Example Learning Curve

```
Step | Val Loss | WER (EWC) | WER (No EWC)
-------------------------------------------
500  | 1.45     | 0.24      | 0.27
1000 | 1.38     | 0.22      | 0.25
1500 | 1.35     | 0.21      | 0.24
```

### Guidelines for Adjustment

- **Limited Compute:** Reduce batch size, Fisher samples, or disable EWC.
- **More Data:** Increase epochs, Fisher samples for better regularization.
- **Faster Adaptation:** Increase learning rate slightly, but monitor for forgetting.

### Trade-offs

- **EWC:** Improves retention of Dutch knowledge, but increases compute and memory.
- **Batch Size:** Higher batch size speeds up training but requires more memory.
- **Learning Rate:** Lower rates improve stability, but slow convergence.

---

## 3. General Recommendations

- Always monitor validation metrics (WER, loss) to guide early stopping.
- Adjust batch size and gradient accumulation based on available GPU memory.
- Use FP16 for efficiency unless numerical instability is observed.
- For reproducibility, set random seeds and document all parameter changes.

---

## 4. Reproducing Results

- Use provided YAML config files for each phase.
- Scripts (`train_dutch_pretrain.py`, `train_german_adaptation.py`) read configs and log metrics.
- For custom hardware, adjust `runpod` section in configs.

---

## 5. References

- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477)
- [Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796)

---

**For further details, consult the config files and training logs in the repository.**