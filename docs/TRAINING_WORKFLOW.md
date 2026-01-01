# ASR Training Workflow Documentation

## Table of Contents
- [1. Pipeline Overview](#1-pipeline-overview)
- [2. Dutch Pre-training](#2-dutch-pre-training)
- [3. German Adaptation (with EWC)](#3-german-adaptation-with-ewc)
- [4. Swiss-German Zero-Shot Transfer](#4-swiss-german-zero-shot-transfer)
- [5. Transfer Learning Approach](#5-transfer-learning-approach)
- [6. Troubleshooting & Common Issues](#6-troubleshooting--common-issues)
- [7. Additional Notes](#7-additional-notes)
- [References](#references)

This guide describes the full pipeline for fine-tuning Wav2Vec2 models for Swiss-German ASR, including Dutch pre-training, German adaptation (with EWC), and Swiss-German zero-shot transfer. It covers step-by-step instructions, command examples, resource requirements, troubleshooting, and the transfer learning strategy.

---

## 1. Pipeline Overview

1. **Dutch Pre-training**: Train Wav2Vec2 on Dutch Common Voice data.
2. **German Adaptation**: Fine-tune the Dutch model on German data, using Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting.
3. **Swiss-German Zero-Shot**: Use the adapted German model for Swiss-German inference or further fine-tuning.

---

## 2. Dutch Pre-training

### Step-by-Step

1. **Prepare Data**: Ensure Dutch Common Voice audio and metadata are available.
2. **Configure Training**: Edit `configs/training/dutch_pretrain.yml` for hyperparameters and paths.
3. **Run Training**: Execute the training script.

### Command Examples

**Local Execution with Docker Compose**

**Warning:** The full Dutch pre-training is resource-intensive. For local testing, use the test command below. Only test the full training on a capable cloud instance. Local testing should be done with a limited dataset as shown in the Important note. Even then, ensure your machine has sufficient resources.

Add these lines to your `docker-compose.yml` under services to create a test service:

```yaml
  dutch-pretrain-test:
    build: .
    command: ["python", "scripts/train_dutch_pretrain.py"]
    volumes:
      - ./data:/app/data
      - ./src:/app/src
      - ./scripts:/app/scripts
      - ./results:/app/results
      - ./configs:/app/configs
      - huggingface-cache:/home/appuser/.cache/huggingface
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - PYTHONPATH=/app
    working_dir: /app
```

Then run the test with:


```bash
docker compose run --rm dutch-pretrain-test
```

**Important!** For local testing, go into `train_dutch_pretrain.py` and set a small limit (e.g., `limit=10`) when preparing the dataset to ensure quick runs. Example:

```python
train_dataset = prepare_dataset(METADATA_FILE, AUDIO_DIR, limit=10)  # Limit to 10 samples for a fast test
```

Once verified, revert the limit change for full training.

**Cloud/RunPod Example**
```bash
# Adjust batch size and workers in dutch_pretrain.yml under 'runpod'
python scripts/train_dutch_pretrain.py
```

### Expected Training Time & Resources

- **Dataset Size**: ~100k samples
- **GPU**: 1x NVIDIA V100/A100 (16GB+ recommended)
- **Time**: 12–24 hours (depends on hardware, batch size, and dataset size)
- **RAM**: 32GB+
- **Disk**: 50GB+ for audio and checkpoints

---

## 3. German Adaptation (with EWC)

**warning:** The same resource considerations as Dutch pre-training apply here. Test locally with a limited dataset first. Here as well, ensure your machine has sufficient resources.

### Step-by-Step

1. **Prepare Data**: Ensure German Common Voice audio and metadata are available.
2. **Configure Training**: Edit `configs/training/german_adaptation.yml` for hyperparameters, EWC settings, and paths.
3. **Run Adaptation**: Execute the adaptation script.

### Command Examples

Create a test service in your `docker-compose.yml`:

```yaml
  german-adapt-test:
    build: .
    command: ["python", "scripts/train_german_adaptation.py"]
    volumes:
      - ./data:/app/data
      - ./src:/app/src
      - ./scripts:/app/scripts
      - ./results:/app/results
      - ./configs:/app/configs
      - huggingface-cache:/home/appuser/.cache/huggingface
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - PYTHONPATH=/app
    working_dir: /app
```

Then run the test with:

**Local Execution with Docker Compose**
```bash
docker compose run --rm german-adapt-test
```

**Cloud/RunPod Example**
```bash
# Adjust batch size and workers in german_adaptation.yml under 'runpod'
python scripts/train_german_adaptation.py
```

### Expected Training Time & Resources

- **Dataset Size**: ~50k samples
- **GPU**: 1x NVIDIA V100/A100 (16GB+ recommended)
- **Time**: 8–16 hours
- **RAM**: 32GB+
- **Disk**: 50GB+

---

## 4. Swiss-German Zero-Shot Transfer

### Step-by-Step

1. **Inference**: Use the German-adapted model for Swiss-German ASR tasks.
2. **Optional Fine-Tuning**: If Swiss-German labeled data is available, fine-tune further.

### Command Example

```python
# Example: Load adapted model and run inference on Swiss-German audio
from src.models.wav2vec2_model import Wav2Vec2Model

model_wrapper = Wav2Vec2Model(model_name="models/pretrained/wav2vec2-german-adapted")
output = model_wrapper.transcribe("path/to/swiss_german_audio.wav")
print(output)
```

---

## 5. Transfer Learning Approach

- **Dutch → German**: The Dutch model is fine-tuned on German data, leveraging shared phonetic and linguistic features.
- **EWC Regularization**: During German adaptation, EWC penalizes changes to parameters important for Dutch, reducing catastrophic forgetting.
- **German → Swiss-German (Zero-Shot)**: The German-adapted model is used directly for Swiss-German, exploiting linguistic proximity.

---

## 6. Troubleshooting & Common Issues

| Issue                          | Solution                                                                                  |
|---------------------------------|------------------------------------------------------------------------------------------|
| **CUDA out of memory**          | Reduce batch size, use gradient accumulation, or switch to a larger GPU.                 |
| **Missing audio files**         | Check metadata paths and ensure all referenced audio files exist.                        |
| **Slow training**               | Increase `num_workers`, use SSD storage, or run on more powerful hardware.               |
| **EWC errors**                  | Ensure Dutch reference data is available and correctly formatted for Fisher estimation.  |
| **Model not improving**         | Tune learning rate, increase patience for early stopping, or check data quality.         |
| **Checkpoint not saving**       | Verify output directory permissions and disk space.                                      |
| **Metric computation errors**   | Ensure labels and predictions are correctly decoded and aligned.                         |

---

## 7. Additional Notes

- **Configuration**: All hyperparameters and paths are managed via YAML files (`configs/training/*.yml`).
- **Logging**: Training logs and metrics are saved in the `results/logs/tensorboard/` directory.
- **Early Stopping**: Enabled by default; adjust patience in YAML configs.
- **Cloud Execution**: For RunPod or similar, adjust batch size and workers in the `runpod` section of the YAML configs.

---

## References

- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477)
- [Elastic Weight Consolidation (EWC)](https://arxiv.org/abs/1612.00796)
- [Common Voice Datasets](https://datacollective.mozillafoundation.org/datasets)
