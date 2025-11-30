import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.config import DATA_DIR, FHNW_SWISS_GERMAN_ROOT
from src.data.splitter import create_splits

if __name__ == "__main__":
    # Validate input file exists (use config paths)
    input_path = FHNW_SWISS_GERMAN_ROOT / "all.tsv"
    if not input_path.exists():
        print(f"❌ Error: Input file not found at {input_path}")
        print("Please download the dataset first!")
        sys.exit(1)
    
    # Create output directory if needed (use config paths)
    output_dir = DATA_DIR / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating train/val/test splits...")
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    try:
        create_splits(
            tsv_path=str(input_path),
            output_dir=str(output_dir)
        )
        print("✓ Data pipeline complete!")
    except Exception as e:
        print(f"❌ Error during data preparation: {e}")
        sys.exit(1)