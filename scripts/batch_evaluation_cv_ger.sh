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
# REMOTE_TEST_PATH and REMOTE_AUDIO_BASE_PATH default to the 1,000-sample
# seeded holdout (test_1000_seed42.tsv, random_state=42, drawn from the
# full 16,197-row test.tsv). This file is reused across every EWC lambda
# condition so all forgetting comparisons share an identical evaluation set.
# Override --test-path explicitly if a different TSV is ever needed (e.g.
# the full test.tsv for a one-off sanity check).
#
# --models has NO default and is required on every invocation. The model
# list changes per run (zero-shot anchor + whichever fine-tuned/EWC variant
# is being scored), so there is no single "right" default to bake in. With
# only a handful of models in play across the EWC grid, typing --models
# explicitly each time is not a meaningful burden and avoids silently
# evaluating the wrong model pair.
#
# Required invocation — always specify --models:
#   ./scripts/batch_evaluation_cv_ger.sh \
#     --models whisper-large-v2 whisper-large-v2-swiss-german-baseline
#
# Override examples:
#   ./scripts/batch_evaluation_cv_ger.sh --models whisper-large-v2 whisper-large-v2-ewc-lambda-3000 --limit 50
#   ./scripts/batch_evaluation_cv_ger.sh --models whisper-large-v2 whisper-large-v2-swiss-german-baseline \
#     --test-path /workspace/data/metadata/german/test.tsv

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

# CV-German-specific defaults
REMOTE_TEST_PATH="/workspace/data/metadata/german/test_1000_seed42.tsv"
REMOTE_AUDIO_BASE_PATH="/workspace/data/raw/cv-corpus-22.0-2025-06-20/de/clips"
REMOTE_OUTPUT_DIR="/workspace/results/metrics/cv-german"
LOCAL_OUTPUT_DIR="results/metrics/cv-german"

# No default model list — --models is required (see header comment).
DEFAULT_MODELS=""
MODELS="$DEFAULT_MODELS"

# Pass-through args not covered by named overrides below (e.g. --limit,
# --experiment-type, --lm-path). Collected as a flat array and appended
# to EVAL_ARGS after all named overrides are resolved.
EXTRA_ARGS=()

# Parse CLI overrides FIRST. EVAL_ARGS is built only after this loop
# completes, so --test-path / --audio-base-path / --models overrides
# take effect instead of producing duplicate, ignored flags.
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            shift
            MODELS=""
            while [[ $# -gt 0 && $1 != --* ]]; do
                MODELS+="$1 "
                shift
            done
            ;;
        --test-path)
            REMOTE_TEST_PATH="$2"
            shift 2
            ;;
        --audio-base-path)
            REMOTE_AUDIO_BASE_PATH="$2"
            shift 2
            ;;
        --output-dir)
            REMOTE_OUTPUT_DIR="$2"
            shift 2
            ;;
        --limit|--experiment-type|--lm-path)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Fail fast, locally, before paying the SSH connection cost.
if [ -z "$MODELS" ]; then
    echo "❌ Error: --models is required (no default model list)."
    echo "   Example: --models whisper-large-v2 whisper-large-v2-swiss-german-baseline"
    exit 1
fi

# Build final eval args now that all overrides are resolved.
EVAL_ARGS=(
    "--models" $MODELS
    "--test-path" "$REMOTE_TEST_PATH"
    "--audio-base-path" "$REMOTE_AUDIO_BASE_PATH"
    "--output-dir" "$REMOTE_OUTPUT_DIR"
    "--experiment-type" "fine-tuned"
    "${EXTRA_ARGS[@]}"
)

EVAL_ARGS_STR="${EVAL_ARGS[*]}"

echo "🚀 Starting German CV 22.0 forgetting benchmark on RunPod..."
echo "   Connecting to: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT}"
echo "   Test set:  ${REMOTE_TEST_PATH}"
echo "   Audio dir: ${REMOTE_AUDIO_BASE_PATH}"
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