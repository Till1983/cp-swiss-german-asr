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
# FIX: Use unquoted heredoc to allow variable expansion
# Note: We need $EVAL_ARGS_STR to expand before sending to remote
ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} bash << ENDSSH
    set -e
    cd /workspace/cp-swiss-german-asr

    echo "📦 Installing requirements (no-cache)..."
    pip install --no-cache-dir -r requirements.txt
    echo "✅ Dependencies installed"
    echo ""

    # Check if KenLM LM file exists, otherwise download
    # Note: download_lm.py saves to src/models/lm/kenLM.arpa
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
    # This bridges the gap between where download_lm.py saves it
    # and where the model registry expects it
    mkdir -p /workspace/models/lm
    if [ ! -L "\$LM_DEST_PATH" ] && [ ! -f "\$LM_DEST_PATH" ]; then
        echo "🔗 Creating symlink for MODEL_REGISTRY compatibility..."
        ln -s "\$LM_SRC_PATH" "\$LM_DEST_PATH"
        echo "   \$LM_DEST_PATH -> \$LM_SRC_PATH"
    fi
    echo ""

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

# FIX: Use correct hardcoded remote path
# The results are always at /workspace/cp-swiss-german-asr/results/metrics/
rsync -avz --progress -e "ssh -p ${REMOTE_PORT}" \
    ${REMOTE_USER}@${REMOTE_HOST}:/workspace/cp-swiss-german-asr/results/metrics/ \
    results/metrics/

echo ""
echo "✅ Results downloaded to: results/metrics/"

# ============================================================================
# Display evaluation summary
# ============================================================================

echo ""
echo "📊 Evaluation Summary"
echo "========================================================================"

LATEST_DIR=$(ls -td results/metrics/*/ 2>/dev/null | head -n 1)

if [ -d "$LATEST_DIR" ]; then
    if command -v jq >/dev/null 2>&1; then
        # Use jq for formatted output of ALL models
        for json_file in "$LATEST_DIR"/*_results.json; do
            if [ -f "$json_file" ]; then
                model_name=$(basename "$json_file" _results.json)
                echo ""
                echo "Model: $model_name"
                echo "----------------------------------------------------------------------"
                jq -r '"  WER:     " + (.overall_wer | tostring) + "%\n  CER:     " + (.overall_cer | tostring) + "%\n  BLEU:    " + (.overall_bleu | tostring) + "\n  Samples: " + (.total_samples | tostring)' "$json_file"
            fi
        done
        echo ""
        echo "========================================================================"
    else
        echo ""
        echo "Results directory: $LATEST_DIR"
        echo ""
        echo "ℹ️  Install jq to view formatted summary:"
        echo "    brew install jq          # macOS"
        echo "    sudo apt-get install jq  # Ubuntu/Debian"
        echo ""
        ls -1 "$LATEST_DIR"/*_results.json 2>/dev/null | while read -r file; do
            echo "  - $(basename "$file")"
        done
    fi
else
    echo "⚠️  No evaluation results found in results/metrics/"
fi

echo ""
echo "🎉 All done! Results saved to: results/metrics/"