from fastapi import APIRouter, HTTPException, status
from src.backend.models import EvaluateRequest, EvaluateResponse
from src.evaluation.evaluator import ASREvaluator

router = APIRouter()

@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_model(request: EvaluateRequest):
    """
    Evaluate a speech recognition model on the test dataset.
    
    Args:
        request: EvaluateRequest containing model and optional limit
        
    Returns:
        EvaluateResponse with evaluation metrics
    """
    try:
        test_set_path = "data/metadata/test.tsv"
        
        # Create evaluator instance with requested model
        evaluator = ASREvaluator(model_name=request.model)
        
        # Load the model
        evaluator.load_model()
        
        # Evaluate the dataset
        results = evaluator.evaluate_dataset(
            metadata_path=test_set_path,
            limit=request.limit
        )
        
        # Return evaluation response
        return EvaluateResponse(
            model=request.model,
            total_samples=results["total_samples"],
            overall_wer=results["overall_wer"],
            per_dialect_wer=results["per_dialect_wer"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during evaluation: {str(e)}"
        )