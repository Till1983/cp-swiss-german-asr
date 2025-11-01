from pydantic import BaseModel
from typing import Dict, Optional, List

class EvaluateRequest(BaseModel):
    """
    Request schema for model evaluation endpoint.
    
    Attributes:
        model: Name of the Whisper model to evaluate (e.g., 'whisper-base', 'whisper-small')
        limit: Optional limit on number of samples to process for testing purposes
    """
    model: str = "whisper-base"
    #audio_files: List[str]
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
    overall_wer: float
    per_dialect_wer: Dict[str, float]