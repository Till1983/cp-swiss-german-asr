#!/bin/bash
# Upload datasets to RunPod persistent volume

set -e  # Exit on error

# Load environment variables from .env file (improved method)
if [ -f .env ]; then
    # Use set -a to auto-export, then source the file
    set -a
    source <(grep -E '^[A-Z_]+=.*' .env)  # Only lines that look like VAR=value
    set +a
elif [ -f ../.env ]; then
    # Try parent directory
    set -a
    source <(grep -E '^[A-Z_]+=.*' ../.env)
    set +a
else
    echo "⚠️  Warning: .env file not found"
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
    echo "   REMOTE_USER=your-runpod-user-id"
    echo "   REMOTE_HOST=ssh.runpod.io"
    echo "   REMOTE_PORT=22"
    echo "   REMOTE_DIR=/workspace/data"
    echo ""
    echo "Debug info:"
    echo "  .env exists: $([ -f .env ] && echo 'yes' || echo 'no')"
    echo "  Current directory: $(pwd)"
    exit 1
fi

# Set defaults for optional variables
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/data}"

echo "📦 Uploading to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}..."
echo "   Using port: ${REMOTE_PORT}"

echo "📦 Uploading Swiss German dataset..."
rsync -avz --progress -e "ssh -p ${REMOTE_PORT}" \
    data/raw/fhnw-swiss-german-corpus/ \
    ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/raw/fhnw-swiss-german-corpus/

echo "📦 Uploading Dutch Common Voice..."
rsync -avz --progress -e "ssh -p ${REMOTE_PORT}" \
    data/raw/cv-corpus-23.0-2025-09-05/ \
    ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/raw/cv-corpus-23.0-2025-09-05/

echo "📦 Uploading German Common Voice..."
rsync -avz --progress -e "ssh -p ${REMOTE_PORT}" \
    data/raw/cv-corpus-22.0-2025-06-20/ \
    ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/raw/cv-corpus-22.0-2025-06-20/

echo "✅ Upload complete!"