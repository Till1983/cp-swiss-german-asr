#!/bin/bash
# Remote Whisper Swiss German training orchestration script
# Launches training in a detached tmux session on RunPod

set -e  # Exit on error

usage() {
    cat << 'EOF'
Usage:
  ./scripts/train_whisper_on_cloud.sh [TRAINING_ARGS...]
  ./scripts/train_whisper_on_cloud.sh --sync-results
  ./scripts/train_whisper_on_cloud.sh --status

Description:
  Thin RunPod wrapper for Whisper Swiss German training.
  Forwards all training args verbatim to:
    python scripts/train_whisper_swiss_german.py --config configs/training/whisper_swiss_german.yml "$@"

Modes:
  --sync-results   Skip launch and sync /workspace/results/ to local results/
                   while excluding fisher artifacts and any models subtree.
  --status         Check whether tmux session whisper-train is currently running.
  -h, --help       Show this help text.

Passthrough flags (handled by train_whisper_swiss_german.py, not this wrapper):
  --smoke_test
  --gradient_checkpointing
  --max_steps
  --ewc_lambda
  --fisher_path
  --theta_star_path
EOF
}

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
SESSION_NAME="whisper-train"

SYNC_RESULTS=false
STATUS_MODE=false
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sync-results)
            SYNC_RESULTS=true
            shift
            ;;
        --status)
            STATUS_MODE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "$SYNC_RESULTS" == true && "$STATUS_MODE" == true ]]; then
    echo "❌ Error: --sync-results and --status cannot be used together."
    exit 1
fi

if [[ "$SYNC_RESULTS" == true ]]; then
    echo "📥 Syncing RunPod results to local machine (excluding fisher and models)..."
    echo "   Connecting to: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT}"
    mkdir -p results/

    rsync -avz --progress -e "ssh -p ${REMOTE_PORT}" \
        --exclude 'fisher/***' \
        --exclude 'models/***' \
        ${REMOTE_USER}@${REMOTE_HOST}:/workspace/results/ \
        results/

    echo ""
    echo "✅ Sync complete. Local results updated in results/"
    exit 0
fi

if [[ "$STATUS_MODE" == true ]]; then
    echo "🔎 Checking remote tmux session status on RunPod..."
    echo "   Connecting to: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT}"

    if ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} "tmux has-session -t ${SESSION_NAME} 2>/dev/null"; then
        echo "✅ Session '${SESSION_NAME}' is running."
        exit 0
    fi

    echo "ℹ️  Session '${SESSION_NAME}' is not running."
    # Preserve non-zero for easy polling in scripts/CI while still printing status text.
    exit 1
fi

REMOTE_ARGS_ESCAPED=""
for arg in "${PASSTHROUGH_ARGS[@]}"; do
    REMOTE_ARGS_ESCAPED+=" $(printf '%q' "$arg")"
done

echo "🚀 Starting Whisper Swiss German training on RunPod..."
echo "   Connecting to: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT}"
echo "   tmux session: ${SESSION_NAME}"
echo ""

ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} << ENDSSH
    set -e
    cd /workspace/cp-swiss-german-asr
    export ENVIRONMENT=runpod
    export PYTHONPATH=/workspace/cp-swiss-german-asr

    if tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
        echo "❌ Error: tmux session '${SESSION_NAME}' already exists."
        echo "   To inspect existing run: tmux attach -t ${SESSION_NAME}"
        echo "   To relaunch from scratch: tmux kill-session -t ${SESSION_NAME}"
        exit 1
    fi

    LOG_DIR="/workspace/results/logs/whisper_swiss_german"
    mkdir -p "\${LOG_DIR}"
    TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
    LOG_FILE="\${LOG_DIR}/run_\${TIMESTAMP}.log"

    TRAIN_CMD="python scripts/train_whisper_swiss_german.py --config configs/training/whisper_swiss_german.yml${REMOTE_ARGS_ESCAPED}"
    TMUX_CMD="cd /workspace/cp-swiss-german-asr && export ENVIRONMENT=runpod && export PYTHONPATH=/workspace/cp-swiss-german-asr && \${TRAIN_CMD} 2>&1 | tee \"\${LOG_FILE}\""

    tmux new-session -d -s ${SESSION_NAME} bash -c "\${TMUX_CMD}"

    echo "✅ Training launched in detached tmux session '${SESSION_NAME}'."
    echo "📄 Log file: \${LOG_FILE}"
    echo ""
    echo "⏳ Waiting 12 seconds, then showing initial tmux output..."
    sleep 12
    echo ""
    echo "----- tmux capture (${SESSION_NAME}) -----"
    tmux capture-pane -p -t ${SESSION_NAME} || true
    echo "----- end capture -----"

    if ! tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
        echo ""
        echo "⚠️  Session '${SESSION_NAME}' already exited within the wait window —"
        echo "    training finished or crashed. Tail of the log:"
        echo "----- tail \${LOG_FILE} -----"
        tail -n 40 "\${LOG_FILE}" 2>/dev/null || echo "    (log file not found)"
        echo "----- end tail -----"
    fi

ENDSSH

echo ""
echo "🧭 To attach: tmux attach -t ${SESSION_NAME}"
echo "🧭 To detach: Ctrl-b then d"
echo "✅ Wrapper finished. Training continues on RunPod."
