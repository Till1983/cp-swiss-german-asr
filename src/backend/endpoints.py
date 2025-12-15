from fastapi import APIRouter, HTTPException, status, Query
from typing import Optional
import json
import re
from pathlib import Path
from src.backend.models import EvaluateRequest, EvaluateResponse
from src.backend.model_cache import get_model_cache
from scripts.evaluate_models import MODEL_REGISTRY
from src.config import RESULTS_DIR

router = APIRouter()

def validate_safe_path_component(component: str, name: str):
    """
    Validate that a path component is safe (alphanumeric, hyphens, underscores).
    Prevents path traversal attacks.
    """
    if not component or not re.match(r"^[a-zA-Z0-9_-]+$", component):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {name}: must contain only alphanumeric characters, hyphens, and underscores"
        )

def get_result_file(model: str, timestamp: Optional[str] = None) -> Path:
    """
    Get the path to the result file for a given model and optional timestamp.
    Handles validation and finding the most recent result if timestamp is not provided.
    """
    metrics_dir = RESULTS_DIR / "metrics"
    validate_safe_path_component(model, "model")
    
    if timestamp:
        validate_safe_path_component(timestamp, "timestamp")
        result_file = metrics_dir / timestamp / f"{model}_results.json"
        if not result_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Results not found for model '{model}' at timestamp '{timestamp}'"
            )
        return result_file
    else:
        if not metrics_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No results directory found"
            )
            
        timestamps = sorted(
            [d for d in metrics_dir.iterdir() if d.is_dir()],
            key=lambda x: x.name,
            reverse=True
        )
        
        for ts_dir in timestamps:
            candidate = ts_dir / f"{model}_results.json"
            if candidate.exists():
                return candidate
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No results found for model '{model}'"
        )


@router.get("/cache/info")
async def cache_info():
    """Return diagnostics for the model cache."""
    cache = get_model_cache()
    return cache.info()


@router.post("/cache/clear")
async def cache_clear():
    """Clear all cached evaluators."""
    cache = get_model_cache()
    cache.clear()
    return {"status": "ok", "message": "Cache cleared"}

@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_model(request: EvaluateRequest):
    """
    Evaluate a speech recognition model on the test dataset.
    
    Args:
        request: EvaluateRequest containing model_type, model, and optional limit
        
    Returns:
        EvaluateResponse with evaluation metrics
    """
    try:
        test_set_path = "data/metadata/test.tsv"
        
        # Fetch evaluator from LRU cache (loads if missing)
        cache = get_model_cache()
        evaluator = cache.get(
            model_type=request.model_type,
            model_name=request.model,
            lm_path=None
        )
        
        # Evaluate the dataset
        results = evaluator.evaluate_dataset(
            metadata_path=test_set_path,
            limit=request.limit
        )
        
        # Return evaluation response
        return EvaluateResponse(
            model=request.model,
            total_samples=results["total_samples"],
            failed_samples=results["failed_samples"],
            overall_wer=results["overall_wer"],
            overall_cer=results["overall_cer"],
            overall_bleu=results["overall_bleu"],
            per_dialect_wer=results["per_dialect_wer"],
            per_dialect_cer=results["per_dialect_cer"],
            per_dialect_bleu=results["per_dialect_bleu"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during evaluation: {str(e)}"
        )
    
@router.get("/models")
async def get_models():
    """
    Get list of available ASR models and their metadata from the registry.
    """
    try:
        # Import MODEL_REGISTRY locally to avoid circular imports or path issues
        # This relies on scripts.evaluate_models being importable from the project root

        models = {}
        for key, config in MODEL_REGISTRY.items():
            models[key] = {
                "type": config.get("type"),
                "display_name": key.replace("-", " ").title(),
                "description": f"Model: {config.get('name')}",
                "supports_lm": "lm_path" in config
            }
        return models
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model registry: {str(e)}"
        )

@router.get("/results")
async def get_results():
    """
    Get list of all available evaluation results.
    """
    metrics_dir = RESULTS_DIR / "metrics"
    if not metrics_dir.exists():
        return []

    results = []
    # Iterate over timestamp directories
    for timestamp_dir in metrics_dir.iterdir():
        if not timestamp_dir.is_dir():
            continue
            
        # Check for result files in the timestamp directory
        for result_file in timestamp_dir.glob("*_results.json"):
            model_name = result_file.name.replace("_results.json", "")
            results.append({
                "model": model_name,
                "timestamp": timestamp_dir.name,
                "path": str(result_file.relative_to(RESULTS_DIR))
            })
    
    # Sort by timestamp descending
    results.sort(key=lambda x: x["timestamp"], reverse=True)
    return results

@router.get("/results/{model}")
async def get_model_results(
    model: str, 
    timestamp: Optional[str] = Query(None)
):
    """
    Get evaluation results for a specific model.
    """
    result_file = get_result_file(model, timestamp)

    try:
        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading result file: {str(e)}"
        )

@router.get("/results/{model}/{dialect}")
async def get_model_dialect_results(
    model: str,
    dialect: str,
    timestamp: Optional[str] = Query(None)
):
    """
    Get dialect-specific evaluation results for a model.
    """
    result_file = get_result_file(model, timestamp)

    try:
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Check if dialect exists in metrics
        results = data.get("results", {})
        per_dialect_wer = results.get("per_dialect_wer", {})
        
        if dialect not in per_dialect_wer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dialect '{dialect}' not found in results for model '{model}'"
            )
            
        # Filter samples
        all_samples = results.get("samples", [])
        dialect_samples = [s for s in all_samples if s.get("dialect") == dialect]
        
        return {
            "samples": dialect_samples,
            "overall_wer": results.get("per_dialect_wer", {}).get(dialect),
            "overall_cer": results.get("per_dialect_cer", {}).get(dialect),
            "overall_bleu": results.get("per_dialect_bleu", {}).get(dialect)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing results: {str(e)}"
        )
