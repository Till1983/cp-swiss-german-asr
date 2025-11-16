#!/bin/bash
# Upload datasets to RunPod persistent volume

set -e  # Exit on error
cd "$(dirname "$0")/.."

###############################################################################
# Robust .env loader (searches up to 3 parent directories, strips CRLF, exports)
###############################################################################

ENV_FILE=""
SEARCH_PATHS=(".env" "../.env" "../../.env" "../../../.env")

for p in "${SEARCH_PATHS[@]}"; do
    if [ -f "$p" ]; then
        ENV_FILE="$p"
        break
    fi
done

if [[ -z "$ENV_FILE" ]]; then
    echo "⚠️  Error: .env file not found in expected locations (searched: ${SEARCH_PATHS[*]})"
    echo "Current directory: $(pwd)"
    exit 1
fi

echo "🔎 Using env file: ${ENV_FILE}"

# Create a temp file with CRLF removed to avoid parsing issues
TMP_ENV="$(mktemp)"
tr -d '\r' < "$ENV_FILE" > "$TMP_ENV"

# Safely export each uppercase VAR=value line
while IFS='=' read -r name value; do
    if [[ "$name" =~ ^[A-Z_]+$ ]]; then
        export "$name"="$value"
    fi
done < <(grep -E '^[A-Z_]+=.*' "$TMP_ENV" || true)

rm -f "$TMP_ENV"


###############################################################################
# Validate required environment variables
###############################################################################

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


###############################################################################
# Defaults for optional variables
###############################################################################

REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/data}"


###############################################################################
# Upload logic (unchanged)
###############################################################################

echo "📦 Uploading to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}..."
echo "   Using port: ${REMOTE_PORT}"

echo "📦 Uploading Swiss German dataset..."
rsync -avz --progress -e "ssh -p ${REMOTE_PORT}" \
    data/raw/fhnw-swiss-german-corpus/ \
    ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/raw/fhnw-swiss-german-corpus/

echo "📦 Uploading Dutch Common Voice..."
rsync -avz --progress -e "ssh -p ${REMOTE_PORT}" \
    data/raw/cv-corpus-23.0-2025-09-05/nl/ \
    ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/raw/cv-corpus-23.0-2025-09-05/nl/

echo "📦 Uploading German Common Voice..."
rsync -avz --progress -e "ssh -p ${REMOTE_PORT}" \
    data/raw/cv-corpus-22.0-2025-06-20/de/ \
    ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/raw/cv-corpus-22.0-2025-06-20/de/

echo "✅ Upload complete!"
