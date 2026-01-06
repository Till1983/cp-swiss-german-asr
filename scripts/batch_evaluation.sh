#!/bin/bash
# batch_evaluation.sh
# Orchestrates remote evaluation of ASR models on RunPod and downloads results

set -e

# Move to project root
cd "$(dirname "$0")/.."

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
    exit 1
fi

REMOTE_PORT="${REMOTE_PORT:-22}"

# Parse CLI arguments for models and other evaluate_models.py args
EVAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            EVAL_ARGS+=("--models")
            shift
            while [[ $# -gt 0 && $1 != --* ]]; do
                EVAL_ARGS+=("$1")
                shift
            done
            ;;
        --test-path|--output-dir|--limit|--experiment-type|--lm-path)
            EVAL_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            EVAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Convert array to space-separated string for SSH transmission
EVAL_ARGS_STR="${EVAL_ARGS[*]}"

echo "🚀 Starting remote batch evaluation on RunPod..."
echo "   Connecting to: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT}"
echo "   Arguments: ${EVAL_ARGS_STR}"
echo ""

# ============================================================================
# Run evaluation on RunPod
# ============================================================================
ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} bash << ENDSSH
    set -e
    cd /workspace/cp-swiss-german-asr

    echo "📦 Installing requirements (no-cache)..."
    apt-get update && apt-get install -y ffmpeg
    apt-get update && apt-get install -y rsync
    pip install --no-cache-dir -r requirements.txt --break-system-packages
    echo "✅ Dependencies installed"
    echo ""

    # Check if KenLM LM file exists, otherwise download
    # Note: download_lm.py saves to src/models/lm/kenLM.arpa (inside project)
    # but MODEL_REGISTRY expects /workspace/models/lm/kenLM.arpa (outside project)
    LM_SRC_PATH="/workspace/cp-swiss-german-asr/src/models/lm/kenLM.arpa"
    LM_DEST_PATH="/workspace/models/lm/kenLM.arpa"
    
    if [ ! -f "\$LM_SRC_PATH" ]; then
        echo "📥 Downloading KenLM language model..."
        python scripts/download_lm.py
        echo "✅ KenLM downloaded to: \$LM_SRC_PATH"
    else
        echo "✅ KenLM found at: \$LM_SRC_PATH"
    fi
    
    # Create symlink for MODEL_REGISTRY compatibility
    mkdir -p /workspace/models/lm
    if [ ! -L "\$LM_DEST_PATH" ] && [ ! -f "\$LM_DEST_PATH" ]; then
        echo "🔗 Creating symlink for MODEL_REGISTRY compatibility..."
        ln -s "\$LM_SRC_PATH" "\$LM_DEST_PATH"
        echo "   \$LM_DEST_PATH -> \$LM_SRC_PATH"
    fi
    echo ""

    # Set environment explicitly
    export ENVIRONMENT=runpod

    echo "🏃 Running evaluation..."
    echo "   Arguments: $EVAL_ARGS_STR"
    python scripts/evaluate_models.py $EVAL_ARGS_STR

    echo ""
    echo "✅ Evaluation finished!"
ENDSSH

echo ""
echo "📥 Downloading evaluation results to local machine..."

# Ensure local directory exists
mkdir -p results/metrics/

# Download from /workspace/results/metrics/ (OUTSIDE project dir)
# This matches RESULTS_DIR in src/config.py for RunPod environment
rsync -avz --progress -e "ssh -p ${REMOTE_PORT}" \
    ${REMOTE_USER}@${REMOTE_HOST}:/workspace/results/metrics/ \
    results/metrics/

echo ""
echo "✅ Results downloaded to: results/metrics/"
echo ""
echo "🎉 Batch evaluation complete!"