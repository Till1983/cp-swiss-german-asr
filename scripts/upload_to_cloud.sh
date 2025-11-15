#!/bin/bash
# Upload datasets to RunPod persistent volume

set -e  # Exit on error

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Use environment variables with fallbacks
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_HOST="${REMOTE_HOST}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/data}"

# Validate required variables
if [ -z "$REMOTE_HOST" ]; then
    echo "❌ Error: REMOTE_HOST not set. Please add it to your .env file"
    echo "   Example: REMOTE_HOST=abc123-xyz789.runpod.io"
    exit 1
fi

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