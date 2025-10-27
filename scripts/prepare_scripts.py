import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data.splitter import create_splits

if __name__ == "__main__":
    # Validate input file exists
    input_path = Path("data/raw/fhnw-swiss-german-corpus/public.tsv")
    if not input_path.exists():
        print(f"❌ Error: Input file not found at {input_path}")
        print("Please download the dataset first!")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = Path("data/metadata")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating train/val/test splits...")
    try:
        create_splits(
            tsv_path=str(input_path),
            output_dir=str(output_dir)
        )
        print("✓ Data pipeline complete!")
    except Exception as e:
        print(f"❌ Error during data preparation: {e}")
        sys.exit(1)