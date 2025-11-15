from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def create_splits(tsv_path, output_dir, data_root=None):
    """Create train/val/test splits stratified by dialect"""
    df = pd.read_csv(tsv_path, sep='\t')

    if data_root is None:
        data_root = Path(tsv_path).parent

    # Add full audio path (now configurable!)
    df['audio_path'] = df['path'].apply(
        lambda x: f"{data_root}/clips/{x}"
    )

    # Split by dialect to ensure representation
    # 70% train, 15% val, 15% test
    train, temp = train_test_split(
        df, test_size=0.3, stratify=df['accent'], random_state=42
    )
    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp['accent'], random_state=42
    )

    # Save splits
    train.to_csv(f"{output_dir}/train.tsv", sep='\t', index=False)
    val.to_csv(f"{output_dir}/val.tsv", sep='\t', index=False)
    test.to_csv(f"{output_dir}/test.tsv", sep='\t', index=False)
    
    # Return stats for validation
    return {
        'train': len(train),
        'val': len(val),
        'test': len(test),
        'dialects': df['accent'].nunique()
    }