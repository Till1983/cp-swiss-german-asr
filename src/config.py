import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Detect environment
IS_RUNPOD = os.path.exists("/workspace") and not os.path.exists("/app")

# Base paths - use different defaults based on environment
if IS_RUNPOD:
    DEFAULT_ROOT = "/workspace"
else:
    DEFAULT_ROOT = "/app"

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", DEFAULT_ROOT))
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", PROJECT_ROOT / "results"))

# Dataset paths
FHNW_SWISS_GERMAN_ROOT = DATA_DIR / "raw" / "fhnw-swiss-german-corpus"
DUTCH_CV_ROOT = DATA_DIR / "raw" / "common-voice-dutch"
GERMAN_CV_ROOT = DATA_DIR / "raw" / "common-voice-german"

# Model cache
CACHE_DIR = MODELS_DIR / "cache"