from pydantic import BaseModel, field_validator
from typing import Optional, Dict, Literal

class EvaluateRequest(BaseModel):
    """
    Request schema for model evaluation endpoint.
    
    Attributes:
        model: Name of the ASR model to evaluate (e.g., 'base' for Whisper, 'facebook/wav2vec2-large-xlsr-53-german' for Wav2Vec2)
        limit: Optional limit on number of samples to process for testing purposes
    """
    model: str
    model_type: Literal["whisper", "wav2vec2", "mms"] = "whisper"  # Default to whisper for backward compatibility
    limit: Optional[int] = None

    @field_validator("limit")
    def validate_limit(cls, value):
        if value is not None and value <= 0:
            raise ValueError("Limit must be a positive integer")
        return value


class EvaluateResponse(BaseModel):
    """
    Response schema containing evaluation results.
    
    Attributes:
        model: Name of the evaluated model
        total_samples: Total number of audio samples processed
        overall_wer: Overall Word Error Rate as a percentage
        per_dialect_wer: WER breakdown by dialect code
    """
    model: str
    total_samples: int
    failed_samples: int
    overall_wer: float
    overall_cer: float
    overall_bleu: float
    per_dialect_wer: Dict[str, float]
    per_dialect_cer: Dict[str, float]
    per_dialect_bleu: Dict[str, float]