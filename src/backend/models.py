from pydantic import BaseModel
from typing import Optional, Dict

class EvaluateRequest(BaseModel):
    """
    Request schema for model evaluation endpoint.
    
    Attributes:
        model: Name of the Whisper model to evaluate (e.g., 'whisper-base', 'whisper-small')
        limit: Optional limit on number of samples to process for testing purposes
    """
    model: str
    model_type: str = "whisper"  # Default to whisper for backward compatibility
    limit: Optional[int] = None


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