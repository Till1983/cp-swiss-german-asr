"""
Convert Common Voice dataset format to match Swiss German structure.
"""
import pandas as pd
from pathlib import Path
import argparse

def prepare_common_voice(
    cv_root: Path,
    output_dir: Path,
    locale: str = "nl"  # Language locale (default: 'nl' for Dutch)
):
    """
    Convert Common Voice TSV to project format.
    
    Args:
        cv_root: Root of Common Voice dataset
        output_dir: Where to save reformatted TSVs
        locale: Language locale (nl for Dutch, de for German)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'dev', 'test']:
        input_file = cv_root / f"{split}.tsv"
        
        if not input_file.exists():
            print(f"⚠️  {split}.tsv not found, skipping...")
            continue
        
        print(f"Processing {split}.tsv...")
        df = pd.read_csv(input_file, sep='\t')
        
        # Add audio_path column
        df['audio_path'] = df['path'].apply(
            lambda x: f"{cv_root}/clips/{x}"
        )
        
        # Rename dev to val for consistency
        output_name = 'val.tsv' if split == 'dev' else f'{split}.tsv'
        output_file = output_dir / output_name
        
        # Save with required columns
        required_cols = [
            'client_id', 'path', 'sentence', 'up_votes', 'down_votes',
            'age', 'gender', 'locale', 'audio_path'
        ]
        
        # Keep only columns that exist
        cols_to_save = [col for col in required_cols if col in df.columns]
        df[cols_to_save].to_csv(output_file, sep='\t', index=False)
        
        print(f"✓ Saved {len(df)} samples to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", required=True, help="Common Voice root dir")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--locale", default="nl", help="Language locale")
    
    args = parser.parse_args()
    
    prepare_common_voice(
        Path(args.cv_root),
        Path(args.output_dir),
        args.locale
    )