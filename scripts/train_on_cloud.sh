#!/bin/bash
# Remote training orchestration script

set -e  # Exit on error

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Validate required environment variables
REQUIRED_VARS=("REMOTE_USER" "REMOTE_HOST")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo "❌ Error: Missing required environment variables in .env file:"
    for var in "${MISSING_VARS[@]}"; do
        echo "   - $var"
    done
    echo ""
    echo "Please add these to your .env file:"
    echo "   REMOTE_USER=root"
    echo "   REMOTE_HOST=your-pod-id.runpod.io"
    echo "   REMOTE_PORT=22"
    echo "   REMOTE_DIR=/workspace/data"
    exit 1
fi

# Set defaults for optional variables
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/data}"

echo "🚀 Starting cloud training job on RunPod..."
echo "   Connecting to: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT}"
echo ""

# Run training ON RUNPOD via SSH
ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} << ENDSSH
    # Everything in this block runs on RunPod
    cd /workspace/cp-swiss-german-asr
    
    # 🆕 Set environment explicitly (recommended over auto-detection)
    export ENVIRONMENT=runpod
    
    # 🆕 OPTIONAL: Explicit path overrides (if not using config.py defaults)
    # export PROJECT_ROOT="/workspace/cp-swiss-german-asr"
    # export DATA_DIR="/workspace/data"
    # export MODELS_DIR="/workspace/models"
    # export RESULTS_DIR="/workspace/results"
    
    # Ensure Python finds the src package
    export PYTHONPATH="/workspace/cp-swiss-german-asr"

    # Run training
    python scripts/train_dutch_pretrain.py \
        --model aware-ai/wav2vec2-large-xlsr-53-german-with-lm \
        --pretrain-data /workspace/data/metadata/dutch/train.tsv \
        --finetune-data /workspace/data/metadata/german/train.tsv \
        --target-data /workspace/data/metadata/train.tsv \
        --output-dir /workspace/models/fine_tuned/wav2vec2-swiss \
        --epochs 10 \
        --batch-size 16 \
        --learning-rate 3e-5
    
    echo "✅ Training complete on RunPod!"
ENDSSH

# Download results FROM RUNPOD TO LAPTOP
echo ""
echo "📥 Downloading model checkpoints to local machine..."
rsync -avz --progress -e "ssh -p ${REMOTE_PORT}" \
    ${REMOTE_USER}@${REMOTE_HOST}:/workspace/models/fine_tuned/ \
    models/fine_tuned/

echo ""
echo "📥 Downloading training results to local machine..."
rsync -avz --progress -e "ssh -p ${REMOTE_PORT}" \
    ${REMOTE_USER}@${REMOTE_HOST}:/workspace/results/ \
    results/

echo ""
echo "🎉 All done! Results saved to:"
echo "   - models/fine_tuned/"
echo "   - results/"