#!/bin/bash
# Upload datasets to RunPod persistent volume

set -e  # Exit on error
cd "$(dirname "$0")/.."

###############################################################################
# Robust .env loader
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
    echo "⚠️  Error: .env file not found"
    exit 1
fi

echo "🔎 Using env file: ${ENV_FILE}"

# Create temp file with CRLF removed
TMP_ENV="$(mktemp)"
tr -d '\r' < "$ENV_FILE" > "$TMP_ENV"

# Export each uppercase VAR=value line
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
    echo "   REMOTE_USER=root"
    echo "   REMOTE_HOST=your-pod-ip"
    echo "   REMOTE_PORT=port"
    echo "   REMOTE_DIR=/workspace/data"
    exit 1
fi

###############################################################################
# Defaults for optional variables
###############################################################################

REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/data}"

###############################################################################
# Create remote directory structure first
###############################################################################

echo "📁 Ensuring remote directory structure exists..."
ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} \
    "mkdir -p ${REMOTE_DIR}/raw/fhnw-swiss-german-corpus ${REMOTE_DIR}/raw/cv-corpus-23.0-2025-09-05/nl ${REMOTE_DIR}/raw/cv-corpus-22.0-2025-06-20/de ${REMOTE_DIR}/metadata"

###############################################################################
# Upload datasets
###############################################################################

echo "📦 Uploading to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}..."
echo "   Using port: ${REMOTE_PORT}"
echo ""

echo "📦 Uploading Swiss German dataset..."
rsync -avz --progress --no-owner --no-group -e "ssh -p ${REMOTE_PORT}" \
    data/raw/fhnw-swiss-german-corpus/ \
    ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/raw/fhnw-swiss-german-corpus/

echo ""
echo "📦 Uploading Dutch Common Voice..."
rsync -avz --progress --no-owner --no-group -e "ssh -p ${REMOTE_PORT}" \
    data/raw/cv-corpus-23.0-2025-09-05/nl/ \
    ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/raw/cv-corpus-23.0-2025-09-05/nl/

echo ""
echo "📦 Uploading German Common Voice..."
rsync -avz --progress --no-owner --no-group -e "ssh -p ${REMOTE_PORT}" \
    data/raw/cv-corpus-22.0-2025-06-20/de/ \
    ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/raw/cv-corpus-22.0-2025-06-20/de/

echo ""
echo "✅ Upload complete!"
echo ""
echo "📊 Verifying upload..."
ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} << 'VERIFY'
echo "Swiss German files: $(ls /workspace/data/raw/fhnw-swiss-german-corpus/clips/ 2>/dev/null | wc -l)"
echo "Dutch files: $(ls /workspace/data/raw/cv-corpus-23.0-2025-09-05/nl/clips/ 2>/dev/null | wc -l)"
echo "German files: $(ls /workspace/data/raw/cv-corpus-22.0-2025-06-20/de/clips/ 2>/dev/null | wc -l)"
VERIFY