import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data.splitter import create_splits

if __name__ == "__main__":
    print("Creating train/val/test splits...")
    create_splits(
        tsv_path="data/raw/fhnw-swiss-german-corpus/public.tsv",
        output_dir="data/metadata"
    )
    print("✓ Data pipeline complete!")