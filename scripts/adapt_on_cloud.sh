#!/bin/bash
# Remote adaptation orchestration script
# Orchestrates the German adaptation phase (EWC) on RunPod

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
    exit 1
fi

# Set defaults for optional variables
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/data}"

echo "🚀 Starting cloud adaptation job on RunPod..."
echo "   Connecting to: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT}"
echo ""

START_TIME=$(date)

# Run adaptation ON RUNPOD via SSH
ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} << ENDSSH
    set -e
    cd /workspace/cp-swiss-german-asr
    export ENVIRONMENT=runpod

    echo "🔍 Checking for Dutch pretrained model..."
    if [ ! -d "/workspace/models/pretrained/wav2vec2-dutch-pretrained" ]; then
        echo "⚠️  ERROR: Could not find Dutch pretrained model at /workspace/models/pretrained/wav2vec2-dutch-pretrained"
        ls -la /workspace/models/pretrained/ || echo "   Directory not found."
        echo "   Adaptation cannot proceed without pretrained model. Exiting."
        exit 1
    else
        echo "✅ Found pretrained model directory."
    fi

    echo "📁 Ensuring adapted model output directory exists..."
    mkdir -p /workspace/models/adapted/wav2vec2-german-adapted

    # Download tokenizer files from Hugging Face
    echo "🌐 Downloading tokenizer files from HuggingFace..."

    wget -O /workspace/models/adapted/wav2vec2-german-adapted/vocab.json \
        https://huggingface.co/aware-ai/wav2vec2-large-xlsr-53-german-with-lm/resolve/main/vocab.json

    wget -O /workspace/models/adapted/wav2vec2-german-adapted/tokenizer_config.json \
        https://huggingface.co/aware-ai/wav2vec2-large-xlsr-53-german-with-lm/resolve/main/tokenizer_config.json

    wget -O /workspace/models/adapted/wav2vec2-german-adapted/special_tokens_map.json \
        https://huggingface.co/aware-ai/wav2vec2-large-xlsr-53-german-with-lm/resolve/main/special_tokens_map.json

    echo "✅ Tokenizer files downloaded."

    ADAPT_START_TIME=\$(date)
    echo "🏃 Starting German Adaptation (EWC) at: \$ADAPT_START_TIME"
    python scripts/train_german_adaptation.py --config configs/training/german_adaptation.yml
    ADAPT_END_TIME=\$(date)
    echo "✅ Adaptation complete on RunPod at: \$ADAPT_END_TIME"
ENDSSH

END_TIME=$(date)

# Download results FROM RUNPOD TO LAPTOP
echo ""
echo "📥 Downloading adapted model checkpoints to local machine..."
rsync -avz --progress -e "ssh -p ${REMOTE_PORT}" \
    ${REMOTE_USER}@${REMOTE_HOST}:/workspace/models/adapted/ \
    models/adapted/

echo ""
echo "📥 Downloading training results/logs to local machine..."
rsync -avz --progress -e "ssh -p ${REMOTE_PORT}" \
    ${REMOTE_USER}@${REMOTE_HOST}:/workspace/results/ \
    results/

echo ""
echo "🎉 All done! Results saved to:"
echo "   - models/adapted/"
echo "   - results/"
echo "🕒 Adaptation job started at: $START_TIME"
echo "🕒 Adaptation job finished at: $END_TIME"
echo "Total duration: $(date -u -d @"$(($(date -d "$END_TIME" +%s) - $(date -d "$START_TIME" +%s)))" +"%H:%M:%S")"