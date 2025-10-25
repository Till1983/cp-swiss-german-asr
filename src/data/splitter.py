import pandas as pd
from sklearn.model_selection import train_test_split

def create_splits(tsv_path, output_dir):
    """Create train/val/test splits stratified by dialect"""
    df = pd.read_csv(tsv_path, sep='\t')

    # Add full audio path
    df['audio_path'] = df['path'].apply(lambda x: f"data/raw/common-voice-swiss-german-corpus/clips/{x}")

    # Split vy dialect to assure representation
    # 70% train, 15% val, 15% test
    train, temp = train_test_split(df, test_size=0.3, stratify=df['accent'])
    val, test = train_test_split(temp, test_size=0.5, stratify=temp['accent'])

    # Save splits
    train.to_csv(f"{output_dir}/train.tsv", sep='\t', index=False)
    val.to_csv(f"{output_dir}/val.tsv", sep='\t', index=False)
    test.to_csv(f"{output_dir}/test.tsv", sep='\t', index=False)