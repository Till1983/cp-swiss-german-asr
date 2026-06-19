#!/bin/bash
# batch_evaluation_cv_ger.sh
# Runs the German Common Voice 22.0 forgetting benchmark on RunPod.
#
# This is a thin wrapper around the shared SSH/rsync logic in batch_evaluation.sh.
# The only differences from the Swiss German evaluation are:
#   - Remote output dir: /workspace/results/metrics/cv-german
#   - Rsync source:      /workspace/results/metrics/cv-german/
#   - Rsync target:      results/metrics/cv-german/
#
# Results land in results/metrics/cv-german/ locally and are intentionally
# separated from results/metrics/ (Swiss German dashboard data).
#
# Default invocation (no-EWC forgetting baseline):
#   ./scripts/batch_evaluation_cv_ger.sh
#
# Override models or add --limit for a quick smoke-check:
#   ./scripts/batch_evaluation_cv_ger.sh --limit 50

set -e

# Move to project root
cd "$(dirname "$0")/.."

# Load environment variables
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

# CV-German-specific paths
REMOTE_TEST_PATH="/workspace/data/metadata/german/test.tsv"
REMOTE_OUTPUT_DIR="/workspace/results/metrics/cv-german"
LOCAL_OUTPUT_DIR="results/metrics/cv-german"

# Default models for the forgetting benchmark.
# Both must be evaluated on the same TSV so evaluate_significance.py
# can pair samples by audio_file for the paired bootstrap test (RQ2).
DEFAULT_MODELS="whisper-large-v2 whisper-large-v2-swiss-german-baseline"

# Build eval args: inject CV-German defaults, then layer in any CLI overrides.
# Caller can override --models, --limit, or --experiment-type if needed.
EVAL_ARGS=(
    "--models" $DEFAULT_MODELS
    "--test-path" "$REMOTE_TEST_PATH"
    "--output-dir" "$REMOTE_OUTPUT_DIR"
    "--experiment-type" "fine-tuned"
)

# Merge caller-supplied overrides (last-writer-wins for duplicate flags is
# handled by argparse on the remote side, so simply appending is safe).
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            # Replace default models entirely
            EVAL_ARGS=("${EVAL_ARGS[@]/--models $DEFAULT_MODELS/}")
            EVAL_ARGS+=("--models")
            shift
            while [[ $# -gt 0 && $1 != --* ]]; do
                EVAL_ARGS+=("$1")
                shift
            done
            ;;
        --limit|--experiment-type|--lm-path)
            EVAL_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            EVAL_ARGS+=("$1")
            shift
            ;;
    esac
done

EVAL_ARGS_STR="${EVAL_ARGS[*]}"

echo "🚀 Starting German CV 22.0 forgetting benchmark on RunPod..."
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
    apt-get update -qq && apt-get install -y ffmpeg rsync -qq
    pip install --default-timeout=600 -r requirements.txt --break-system-packages -q
    echo "✅ Dependencies installed"
    echo ""

    # KenLM — required by some registry entries; skip if not needed for this run
    LM_SRC_PATH="/workspace/cp-swiss-german-asr/src/models/lm/kenLM.arpa"
    LM_DEST_PATH="/workspace/models/lm/kenLM.arpa"

    if [ ! -f "\$LM_SRC_PATH" ]; then
        echo "📥 Downloading KenLM language model..."
        python scripts/download_lm.py
    else
        echo "✅ KenLM found at: \$LM_SRC_PATH"
    fi

    mkdir -p /workspace/models/lm
    if [ ! -L "\$LM_DEST_PATH" ] && [ ! -f "\$LM_DEST_PATH" ]; then
        ln -s "\$LM_SRC_PATH" "\$LM_DEST_PATH"
    fi

    export ENVIRONMENT=runpod

    echo "🏃 Running evaluation..."
    echo "   Arguments: $EVAL_ARGS_STR"
    python scripts/evaluate_models.py $EVAL_ARGS_STR

    echo ""
    echo "✅ Evaluation finished!"
ENDSSH

echo ""
echo "📥 Downloading results to local machine..."

mkdir -p "$LOCAL_OUTPUT_DIR"

rsync -avz --progress -e "ssh -p ${REMOTE_PORT}" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_OUTPUT_DIR}/" \
    "${LOCAL_OUTPUT_DIR}/"

echo ""
echo "✅ Results downloaded to: ${LOCAL_OUTPUT_DIR}/"
echo ""
echo "🎉 German CV 22.0 forgetting benchmark complete!"
echo ""
echo "Next step — run significance test locally:"
echo "  python scripts/evaluate_significance.py \\"
echo "    --results-a ${LOCAL_OUTPUT_DIR}/<timestamp>/whisper-large-v2_results.json \\"
echo "    --results-b ${LOCAL_OUTPUT_DIR}/<timestamp>/whisper-large-v2-swiss-german-baseline_results.json \\"
echo "    --metric wer --n-bootstrap 10000 --seed 42"
