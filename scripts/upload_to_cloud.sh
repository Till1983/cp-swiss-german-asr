#!/bin/bash
# Upload datasets to RunPod persistent volume

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