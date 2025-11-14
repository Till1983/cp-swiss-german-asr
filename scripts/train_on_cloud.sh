#!/bin/bash
# Remote training orchestration script

set -e  # Exit on error

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "🚀 Starting cloud training job..."

# Set environment for cloud
export PROJECT_ROOT="/workspace"
export DATA_DIR="/workspace/data"
export MODELS_DIR="/workspace/models"
export RESULTS_DIR="/workspace/results"

# Run training with Dutch→German→Swiss transfer
python scripts/train_wav2vec2_model.py \
    --model wav2vec2-german \
    --pretrain-data /workspace/data/raw/common-voice-dutch \
    --finetune-data /workspace/data/raw/common-voice-german \
    --target-data /workspace/data/metadata/train.tsv \
    --output-dir /workspace/models/fine_tuned/wav2vec2-swiss \
    --epochs 10 \
    --batch-size 16 \
    --learning-rate 3e-5

echo "✅ Training complete! Downloading checkpoints..."

# Download results back to local
rsync -avz --progress \
    ${REMOTE_USER}@${REMOTE_HOST}:/workspace/models/fine_tuned/ \
    models/fine_tuned/

rsync -avz --progress \
    ${REMOTE_USER}@${REMOTE_HOST}:/workspace/results/ \
    results/