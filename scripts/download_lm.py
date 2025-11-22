import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LM_DIR = PROJECT_ROOT / "src" / "models" / "lm"
LM_FILE_NAME = "kenLM.arpa"
DESTINATION_PATH = LM_DIR / LM_FILE_NAME

# Model details
REPO_ID = "aware-ai/wav2vec2-large-xlsr-53-german-with-lm"
# Based on the repo structure, the file is inside a "language_model" folder
REMOTE_FILENAME = "language_model/kenLM.arpa"

def download_german_lm():
    """
    Downloads the KenLM arpa file from Hugging Face with caching.
    """
    print(f"Checking for Language Model: {LM_FILE_NAME}...")
    
    # Ensure directory exists
    LM_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # This function handles caching automatically. 
        # If the file is already in ~/.cache/huggingface, it won't download it again.
        cached_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=REMOTE_FILENAME,
            repo_type="model"
        )
        print(f"✓ File located in HF Cache: {cached_path}")

        # We want a stable path for our registry to point to (src/models/lm/kenLM.arpa)
        # We copy it there (or symlink if on Linux/Mac to save space)
        if DESTINATION_PATH.exists():
            # Check if it's the same file size to avoid unnecessary overwrites
            if DESTINATION_PATH.stat().st_size == Path(cached_path).stat().st_size:
                print(f"✓ Local file already exists and matches cache: {DESTINATION_PATH}")
                return str(DESTINATION_PATH)
            else:
                print(f"Updating local file at {DESTINATION_PATH}...")
        
        # Copy file to project structure
        shutil.copy2(cached_path, DESTINATION_PATH)
        print(f"✓ Successfully deployed LM to: {DESTINATION_PATH}")
        return str(DESTINATION_PATH)

    except Exception as e:
        print(f"❌ Failed to download LM: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_german_lm()