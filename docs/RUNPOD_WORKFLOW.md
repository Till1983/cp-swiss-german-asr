# RunPod Training Workflow

## Table of Contents
- [One-Time Setup](#one-time-setup)
- [Data Upload (First Time Only)](#data-upload-first-time-only)
- [Prepare Datasets (First Time Only)](#prepare-datasets-first-time-only)
- [Downloading the KenLM ARPA Model on RunPod](#downloading-the-kenlm-arpa-model-on-runpod)
- [Run Training](#run-training)
- [Batch Evaluation on RunPod](#batch-evaluation-on-runpod)
- [RunPod Error Analysis](#runpod-error-analysis)
- [Cost Management](#cost-management)
- [See Also](#see-also)

## One-Time Setup
1. Create RunPod account
2. Create 100GB network volume
3. Deploy RTX 3090 Secure Cloud pod
4. Add SSH key to RunPod settings
5. Update `.env` with pod connection info

## Data Upload (First Time Only)
```bash
# On your laptop
./upload_to_cloud.sh
```

## Prepare Datasets (First Time Only)
```bash
# SSH into RunPod
ssh root@your-pod-id.runpod.io

cd /workspace/cp-swiss-german-asr

# Prepare Dutch metadata
python scripts/prepare_common_voice.py \
    --cv-root /workspace/data/raw/cv-corpus-23.0-2025-09-05/nl \
    --locale nl \
    --output-dir /workspace/data/metadata/dutch

# Prepare German metadata
python scripts/prepare_common_voice.py \
    --cv-root /workspace/data/raw/cv-corpus-22.0-2025-06-20/de \
    --locale de \
    --output-dir /workspace/data/metadata/german

# Prepare Swiss German splits
python scripts/prepare_scripts.py
```

## Downloading the KenLM ARPA Model on RunPod

To download the KenLM ARPA file from HuggingFace directly in your RunPod environment, run the following command in the RunPod console:

```bash
python scripts/download_lm.py
```

This will create the `kenLM.arpa` file in `src/models/lm/` within your project directory.

**Tip:**  
Ensure Python dependencies are installed (`pip install -r requirements.txt`) before running the script for the first time.

## Run Training
```bash
# On your laptop (triggers remote training)
./scripts/train_on_cloud.sh
```

## Batch Evaluation on RunPod
```bash
# On your laptop (runs evaluate_models.py remotely, downloads metrics)
./scripts/batch_evaluation.sh \
    --models whisper-large-v3 wav2vec2-1b-german-cv11 \
    --limit 25
```

- Reads `REMOTE_USER`, `REMOTE_HOST` (and optional `REMOTE_PORT`, default 22) from `.env`.
- Installs requirements and ensures `kenLM.arpa` is present/symlinked on the pod.
- Uploads args to `scripts/evaluate_models.py` on RunPod and syncs results back to `results/metrics/` locally.

## RunPod Error Analysis
```bash
# On your laptop (analyzes downloaded metrics on RunPod, syncs outputs)
./scripts/runpod_analyze_errors.sh
```

- Uses the same `.env` values for SSH; expects metrics in `/workspace/results/metrics/` on the pod (e.g., from batch evaluation).
- Runs `scripts/analyze_errors.py` remotely with default settings (top 10% worst samples) and downloads outputs to `results/error_analysis/` locally.
- Logs and artifacts are timestamped under `results/error_analysis/` both remotely and after sync.

## Cost Management
- **While working:** Keep pod running (~$0.47/hr)
- **Between sessions:** Terminate pod (data persists on volume)
- **Storage cost:** ~$0.01/hr for network volume (always running)

## See Also
   - [RunPod Pod Persistence Guide](RUNPOD_POD_PERSISTENCE.md) - Understanding storage and package management on RunPod pods.