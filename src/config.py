"""
Configuration module for Swiss German ASR project.

Supports multiple environments:
- local: Laptop development (macOS/Windows/Linux)
- runpod: Cloud GPU environment with network volume
- ci: Continuous integration (future use)

Environment Selection:
1. Set ENVIRONMENT variable explicitly (recommended)
2. Auto-detect based on filesystem (deprecated, backwards compatibility)

Example:
    export ENVIRONMENT=runpod  # Explicit (recommended)
    # or let it auto-detect (backwards compatible)
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================

def _detect_environment_legacy():
    """
    DEPRECATED: Auto-detect environment based on filesystem.
    
    This is kept for backwards compatibility but should not be relied upon.
    Set ENVIRONMENT variable explicitly instead.
    """
    if os.path.exists("/workspace") and not os.path.exists("/app"):
        logger.warning(
            "⚠️  DEPRECATION WARNING: Auto-detecting RunPod environment. "
            "Please set ENVIRONMENT=runpod explicitly in the future."
        )
        return "runpod"
    return "local"


# Get environment: explicit setting takes precedence over auto-detection
ENVIRONMENT = os.getenv("ENVIRONMENT")

if ENVIRONMENT is None:
    ENVIRONMENT = _detect_environment_legacy()
    IS_AUTO_DETECTED = True
else:
    IS_AUTO_DETECTED = False

# Validate environment
VALID_ENVIRONMENTS = ["local", "runpod", "ci"]
if ENVIRONMENT not in VALID_ENVIRONMENTS:
    logger.warning(
        f"Unknown ENVIRONMENT='{ENVIRONMENT}', falling back to 'local'. "
        f"Valid options: {VALID_ENVIRONMENTS}"
    )
    ENVIRONMENT = "local"

# Legacy compatibility flag (keeping your original variable for any code that uses it)
IS_RUNPOD = (ENVIRONMENT == "runpod")

# ============================================================================
# ENVIRONMENT-SPECIFIC DEFAULTS
# ============================================================================

ENV_DEFAULTS = {
    "local": {
        "PROJECT_ROOT": "/app",
        "DATA_DIR": "/app/data",
        "MODELS_DIR": "/app/models",
        "RESULTS_DIR": "/app/results",
    },
    "runpod": {
        "PROJECT_ROOT": "/workspace/cp-swiss-german-asr",
        "DATA_DIR": "/workspace/data",      # Outside project directory
        "MODELS_DIR": "/workspace/models",  # Outside project directory
        "RESULTS_DIR": "/workspace/results", # Outside project directory
    },
    "ci": {
        "PROJECT_ROOT": "/tmp/cp-swiss-german-asr",
        "DATA_DIR": "/tmp/data",
        "MODELS_DIR": "/tmp/models",
        "RESULTS_DIR": "/tmp/results",
    }
}

defaults = ENV_DEFAULTS[ENVIRONMENT]

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Base paths - environment variables override defaults
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", defaults["PROJECT_ROOT"]))
DATA_DIR = Path(os.getenv("DATA_DIR", defaults["DATA_DIR"]))
MODELS_DIR = Path(os.getenv("MODELS_DIR", defaults["MODELS_DIR"]))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", defaults["RESULTS_DIR"]))

# Dataset paths (relative to DATA_DIR)
FHNW_SWISS_GERMAN_ROOT = DATA_DIR / "raw" / "fhnw-swiss-german-corpus"
DUTCH_CV_ROOT = DATA_DIR / "raw" / "cv-corpus-23.0-2025-09-05" / "nl"
GERMAN_CV_ROOT = DATA_DIR / "raw" / "cv-corpus-22.0-2025-06-20" / "de"
# Model cache
CACHE_DIR = MODELS_DIR / "cache"

# ============================================================================
# VALIDATION AND LOGGING
# ============================================================================

def validate_paths(strict: bool = False):
    """
    Validate that critical paths exist.
    
    Args:
        strict: If True, raise exception on missing paths.
                If False, only log warnings (useful for testing/imports).
    
    Raises:
        RuntimeError: If strict=True and critical paths don't exist.
    """
    critical_paths = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "DATA_DIR": DATA_DIR,
    }
    
    missing_paths = []
    for name, path in critical_paths.items():
        if not path.exists():
            missing_paths.append(f"{name}={path}")
    
    if missing_paths:
        msg = (
            f"Missing paths in environment '{ENVIRONMENT}':\n  " + 
            "\n  ".join(missing_paths)
        )
        if strict:
            raise RuntimeError(
                f"{msg}\n\n"
                "Please verify your configuration:\n"
                f"  1. Check ENVIRONMENT={ENVIRONMENT} is correct\n"
                f"  2. Check your .env file or environment variables\n"
                f"  3. Ensure paths exist or can be created"
            )
        else:
            logger.debug(msg)


def log_configuration():
    """Log active configuration for debugging and reproducibility."""
    detection_method = "auto-detected (deprecated)" if IS_AUTO_DETECTED else "explicit"
    
    logger.info("=" * 70)
    logger.info(f"Environment: {ENVIRONMENT} ({detection_method})")
    logger.info("-" * 70)
    logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")
    logger.info(f"DATA_DIR:     {DATA_DIR}")
    logger.info(f"MODELS_DIR:   {MODELS_DIR}")
    logger.info(f"RESULTS_DIR:  {RESULTS_DIR}")
    logger.info(f"CACHE_DIR:    {CACHE_DIR}")
    logger.info("=" * 70)
    
    if IS_AUTO_DETECTED:
        logger.warning(
            "💡 TIP: Set ENVIRONMENT variable explicitly:\n"
            f"   export ENVIRONMENT={ENVIRONMENT}"
        )


# Validate on import (non-strict to allow imports during testing)
validate_paths(strict=False)

# Log configuration
log_configuration()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config_summary() -> dict:
    """
    Get current configuration as a dictionary.
    
    Useful for logging to results files for reproducibility.
    
    Returns:
        dict: Configuration summary
    """
    return {
        "environment": ENVIRONMENT,
        "is_auto_detected": IS_AUTO_DETECTED,
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "models_dir": str(MODELS_DIR),
        "results_dir": str(RESULTS_DIR),
        "cache_dir": str(CACHE_DIR),
    }


def create_directories():
    """
    Create all configured directories if they don't exist.
    
    Useful for initialization scripts.
    """
    directories = [
        PROJECT_ROOT,
        DATA_DIR,
        DATA_DIR / "raw",
        DATA_DIR / "processed",
        DATA_DIR / "metadata",
        MODELS_DIR,
        MODELS_DIR / "cache",
        MODELS_DIR / "pretrained",
        MODELS_DIR / "fine_tuned",
        RESULTS_DIR,
        RESULTS_DIR / "metrics",
        RESULTS_DIR / "figures",
        RESULTS_DIR / "analysis",
        RESULTS_DIR / "logs",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")
    
    logger.info(f"Created/verified {len(directories)} directories")


if __name__ == "__main__":
    # If run directly, print config and validate strictly
    print("\n🔍 Configuration Check\n")
    log_configuration()
    print("\n✓ Validating paths (strict mode)...\n")
    validate_paths(strict=True)
    print("✅ All paths valid!\n")