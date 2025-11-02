from fastapi import FastAPI, HTTPException
import whisper
import torch
import logging
from src.backend import endpoints

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/load-model")
async def load_model():
    try:
        logger.info("Starting model loading process...")
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        model = whisper.load_model("base").to(device)
        logger.info("Model loaded successfully")
        
        return {
            "status": "Model loaded successfully",
            "model": "whisper-base",
            "device": device
        }
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

app.include_router(endpoints.router, prefix="/api", tags=["evaluation"])