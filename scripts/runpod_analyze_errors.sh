#!/bin/bash
#
# RunPod Error Analysis Pipeline
# Orchestrates error analysis on RunPod and downloads results locally.
#

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set UTF-8 encoding
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# Load environment variables from .env file if present
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
    echo -e "\033[0;31m[ERROR] Missing required environment variables in .env file:\033[0m"
    for var in "${MISSING_VARS[@]}"; do
        echo "   - $var"
    done
    exit 1
fi

REMOTE_PORT="${REMOTE_PORT:-22}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ============================================================================
# REMOTE EXECUTION
# ============================================================================

echo -e "${GREEN}>> Starting error analysis on RunPod...${NC}"
echo "   Connecting to: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT}"
echo ""

# Execute analysis pipeline on RunPod
ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} bash << 'ENDSSH'
    set -e
    export LANG=en_US.UTF-8
    export LC_ALL=en_US.UTF-8
    
    cd /workspace/cp-swiss-german-asr

    echo "📦 Installing requirements (no-cache)..."
    pip install --no-cache-dir -r requirements_blackwell.txt --break-system-packages
    
    # Define paths
    RESULTS_DIR="/workspace/results"
    METRICS_DIR="${RESULTS_DIR}/metrics"
    ANALYSIS_DIR="${RESULTS_DIR}/error_analysis"
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    CURRENT_ANALYSIS_DIR="${ANALYSIS_DIR}/${TIMESTAMP}"
    
    # Create directories
    mkdir -p "${ANALYSIS_DIR}"
    LOG_FILE="${ANALYSIS_DIR}/pipeline_${TIMESTAMP}.log"
    
    # Logging function
    log_message() {
        local msg="$1"
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo -e "[${timestamp}] ${msg}" | tee -a "${LOG_FILE}"
    }
    
    # Colors
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    RED='\033[0;31m'
    NC='\033[0m'
    
    log_message "${GREEN}========================================${NC}"
    log_message "${GREEN}   RUNPOD ERROR ANALYSIS PIPELINE     ${NC}"
    log_message "${GREEN}========================================${NC}"
    log_message "Log file: ${LOG_FILE}"
    log_message "Results Dir: ${RESULTS_DIR}"
    
    # ========================================================================
    # STEP 1: Environment Validation
    # ========================================================================
    log_message ""
    log_message "${YELLOW}STEP 1/4: Environment Validation${NC}"
    
    # Check Metrics Directory
    if [ ! -d "${METRICS_DIR}" ]; then
        log_message "${RED}[CRITICAL ERROR] Metrics directory not found at ${METRICS_DIR}${NC}"
        exit 1
    fi
    
    # Count result files
    RESULT_FILES=$(find "${METRICS_DIR}" -name "*_results.json" | wc -l)
    log_message "Found ${RESULT_FILES} result files in ${METRICS_DIR}"
    
    if [ "${RESULT_FILES}" -eq 0 ]; then
        log_message "${RED}[CRITICAL ERROR] No *_results.json files found to analyze.${NC}"
        exit 1
    fi
    
    # Verify Python
    PYTHON_VER=$(python --version 2>&1)
    log_message "Python Environment: ${PYTHON_VER}"
    
    # Check script existence
    if [ ! -f "scripts/analyze_errors.py" ]; then
        log_message "${RED}[CRITICAL ERROR] scripts/analyze_errors.py not found.${NC}"
        exit 1
    fi
    
    log_message "${GREEN}[OK] Environment validation passed.${NC}"
    
    # ========================================================================
    # STEP 2: Directory Setup
    # ========================================================================
    log_message ""
    log_message "${YELLOW}STEP 2/4: Directory Setup${NC}"
    
    mkdir -p "${CURRENT_ANALYSIS_DIR}"
    log_message "Created analysis output directory: ${CURRENT_ANALYSIS_DIR}"
    
    # List existing analysis directories (last 5)
    log_message "Existing analysis runs:"
    ls -d "${ANALYSIS_DIR}"/*/ 2>/dev/null | tail -n 5 | sed 's/^/  - /' || echo "  (None)"
    
    # ========================================================================
    # STEP 3: Run Analysis
    # ========================================================================
    log_message ""
    log_message "${YELLOW}STEP 3/4: Running Error Analysis${NC}"
    log_message "This may take 2-5 minutes depending on dataset size..."
    log_message "Progress updates will appear below:"
    
    # Keepalive background process
    (
        while sleep 30; do
            echo "... still processing"
        done
    ) &
    KEEPALIVE_PID=$!
    
    # Run the python script
    set +e # Temporarily disable exit-on-error to capture python failure
    python scripts/analyze_errors.py \
        --input_dir "${METRICS_DIR}" \
        --output_dir "${CURRENT_ANALYSIS_DIR}" \
        --top_percent 0.10 \
        2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    set -e # Re-enable
    
    # Kill keepalive
    if [ -n "$KEEPALIVE_PID" ] && kill -0 "$KEEPALIVE_PID" 2>/dev/null; then
        kill "$KEEPALIVE_PID" 2>/dev/null || true
        wait "$KEEPALIVE_PID" 2>/dev/null || true
    fi
    
    if [ $EXIT_CODE -ne 0 ]; then
        log_message "${RED}[WARNING] Analysis script exited with code ${EXIT_CODE}${NC}"
        log_message "Check log for details. Proceeding to verification..."
    else
        log_message "${GREEN}[OK] Analysis script finished successfully.${NC}"
    fi
    
    # ========================================================================
    # STEP 4: Verify Outputs
    # ========================================================================
    log_message ""
    log_message "${YELLOW}STEP 4/4: Verifying Outputs${NC}"
    
    JSON_COUNT=$(find "${CURRENT_ANALYSIS_DIR}" -name "*.json" | wc -l)
    CSV_COUNT=$(find "${CURRENT_ANALYSIS_DIR}" -name "*.csv" | wc -l)
    SUMMARY_EXISTS=false
    
    if [ -f "${CURRENT_ANALYSIS_DIR}/model_comparison_summary.json" ]; then
        SUMMARY_EXISTS=true
    fi
    
    log_message "Generated Files:"
    log_message "  - JSON files: ${JSON_COUNT}"
    log_message "  - CSV files:  ${CSV_COUNT}"
    
    if [ "$SUMMARY_EXISTS" = true ]; then
        log_message "${GREEN}[OK] model_comparison_summary.json found.${NC}"
    else
        log_message "${RED}[ERROR] model_comparison_summary.json MISSING.${NC}"
    fi
    
    log_message "File listing:"
    ls -lh "${CURRENT_ANALYSIS_DIR}" 2>/dev/null | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}' | tee -a "$LOG_FILE"
    
    # ========================================================================
    # Summary Report
    # ========================================================================
    log_message ""
    log_message "${GREEN}========================================${NC}"
    log_message "${GREEN}   PIPELINE COMPLETE                  ${NC}"
    log_message "${GREEN}========================================${NC}"
    log_message "Total duration: ${SECONDS}s"
    log_message "Results saved to: ${CURRENT_ANALYSIS_DIR}"
    log_message "Log file: ${LOG_FILE}"
    
    if [ "$SUMMARY_EXISTS" = true ]; then
        exit 0
    else
        log_message "${RED}Pipeline finished but critical outputs are missing.${NC}"
        exit 1
    fi
ENDSSH

# ============================================================================
# DOWNLOAD RESULTS
# ============================================================================

echo ""
echo -e "${GREEN}>> Downloading analysis results to local machine...${NC}"

# Create local directory
mkdir -p results/error_analysis/

# Sync results back
rsync -avz --progress -e "ssh -p ${REMOTE_PORT}" \
    ${REMOTE_USER}@${REMOTE_HOST}:/workspace/results/error_analysis/ \
    results/error_analysis/

echo ""
echo -e "${GREEN}>> Analysis pipeline complete!${NC}"
echo "   Local results: results/error_analysis/"
echo "   View latest: results/error_analysis/\$(ls -t results/error_analysis/ | head -1)"