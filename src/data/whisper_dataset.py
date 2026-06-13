"""FHNW Swiss German dataset wiring for Whisper fine-tuning.

Produces collator-ready examples (``input_features`` + ``labels``) from the
FHNW metadata TSVs and audio clips. Audio paths are reconstructed from
``clips_dir`` + a cleaned ``path`` column rather than trusting the precomputed
``audio_path`` column, because that column is written with the local Docker
prefix (``/app/...``) and would be wrong under ``ENVIRONMENT=runpod``.
"""

import csv
import logging
from pathlib import Path
from typing import List, Optional

import librosa
import pandas as pd
import torch

from src.data.whisper_collator import prepare_whisper_example

logger = logging.getLogger(__name__)


def clean_path(p: str) -> str:
    """Strip embedded newline/CR/tab characters from a metadata path entry."""
    return str(p).strip().replace("\n", "").replace("\r", "").replace("\t", "")


def load_metadata_df(
    metadata_path,
    subset_size: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load a FHNW metadata TSV, optionally sampling a fixed subset."""
    df = pd.read_csv(
        metadata_path, sep="\t", low_memory=False, encoding="utf-8", quoting=csv.QUOTE_NONE
    )
    if subset_size is not None and subset_size < len(df):
        df = df.sample(n=subset_size, random_state=seed).reset_index(drop=True)
    return df


class WhisperSpeechDataset(torch.utils.data.Dataset):
    """Lazily load audio and produce Whisper ``input_features`` + ``labels``.

    Each item is a dict ready for ``DataCollatorSpeechSeq2SeqWithPadding``.
    The per-row dialect labels are also exposed via :attr:`dialects` (aligned
    with item order) so eval predictions can be grouped per canton without
    threading the column through the collator.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        clips_dir,
        processor,
        sampling_rate: int = 16000,
        dialect_column: str = "accent",
        text_column: str = "sentence",
        path_column: str = "path",
    ):
        self.df = metadata_df.reset_index(drop=True)
        self.clips_dir = Path(clips_dir)
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.dialect_column = dialect_column
        self.text_column = text_column
        self.path_column = path_column

    @property
    def dialects(self) -> List[str]:
        if self.dialect_column in self.df.columns:
            return self.df[self.dialect_column].astype(str).tolist()
        return ["unknown"] * len(self.df)

    @property
    def references(self) -> List[str]:
        return self.df[self.text_column].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio_path = self.clips_dir / clean_path(row[self.path_column])
        audio, _ = librosa.load(str(audio_path), sr=self.sampling_rate)
        return prepare_whisper_example(
            self.processor,
            audio,
            str(row[self.text_column]),
            sampling_rate=self.sampling_rate,
        )
