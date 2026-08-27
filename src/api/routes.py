"""API routes for TensorVision AI."""

import os
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import io

from src.inference.engine import InferenceEngine
from src.config import get_config

router = APIRouter()

engine: Optional[InferenceEngine] = None


class PredictionResponse(BaseModel):
    """Single prediction response."""
    prediction: str
    confidence: float
    model_version: str
    class_index: int
    above_threshold: bool
    top_k: Optional[List[dict]] = None
    all_probabilities: Optional[dict] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total: int


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_version: str
    num_classes: int
    class_names: List[str]
    input_shape: List[int]
    output_shape: List[int]
    total_params: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


def get_engine() -> InferenceEngine:
    """Dependency to get the inference engine."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return engine


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if engine is not None else "unhealthy",
        model_loaded=engine is not None,
        model_version=engine.model_version if engine else None
    )


@router.get("/model", response_model=ModelInfoResponse)
async def get_model_info(eng: InferenceEngine = Depends(get_engine)):
    """Get model information."""
    info = eng.get_model_info()
    return ModelInfoResponse(**info)


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    top_k: Optional[int] = Form(None),
    confidence_threshold: Optional[float] = Form(None),
    return_probabilities: bool = Form(True),
    eng: InferenceEngine = Depends(get_engine)
):
    """Predict class for a single uploaded image."""
    await _validate_file(file)
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = image.convert('RGB')
    
    original_top_k = eng.top_k
    original_threshold = eng.confidence_threshold
    
    if top_k is not None:
        eng.top_k = top_k
    if confidence_threshold is not None:
        eng.confidence_threshold = confidence_threshold
    
    try:
        result = eng.predict(image, return_probabilities=return_probabilities)
    finally:
        eng.top_k = original_top_k
        eng.confidence_threshold = original_threshold
    
    return PredictionResponse(**result)


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    files: List[UploadFile] = File(...),
    top_k: Optional[int] = Form(None),
    confidence_threshold: Optional[float] = Form(None),
    return_probabilities: bool = Form(True),
    eng: InferenceEngine = Depends(get_engine)
):
    """Predict classes for multiple uploaded images."""
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 files per batch")
    
    images = []
    for file in files:
        await _validate_file(file)
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.convert('RGB')
        images.append(image)
    
    original_top_k = eng.top_k
    original_threshold = eng.confidence_threshold
    
    if top_k is not None:
        eng.top_k = top_k
    if confidence_threshold is not None:
        eng.confidence_threshold = confidence_threshold
    
    try:
        results = eng.predict_batch(images, return_probabilities=return_probabilities)
    finally:
        eng.top_k = original_top_k
        eng.confidence_threshold = original_threshold
    
    return BatchPredictionResponse(
        predictions=[PredictionResponse(**r) for r in results],
        total=len(results)
    )


async def _validate_file(file: UploadFile):
    """Validate uploaded file."""
    api_config = get_config().api
    
    if file.size and file.size > api_config.get('max_file_size', 10485760):
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {api_config.get('max_file_size', 10485760)} bytes"
        )
    
    allowed_extensions = api_config.get('allowed_extensions', ['.jpg', '.jpeg', '.png', '.bmp', '.webp'])
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {allowed_extensions}"
        )
    
    try:
        contents = await file.read()
        await file.seek(0)
        Image.open(io.BytesIO(contents)).verify()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")


def initialize_engine(
    model_path: str,
    class_names: Optional[List[str]] = None
) -> InferenceEngine:
    """Initialize the global inference engine."""
    global engine
    engine = create_inference_engine(model_path, class_names)
    return engine