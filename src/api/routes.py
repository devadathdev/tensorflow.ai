"""API routes for TensorVision AI."""

from pathlib import Path
from typing import List, Optional
import io

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from pydantic import BaseModel, Field
from PIL import Image

from src.inference.engine import InferenceEngine, create_inference_engine
from src.config import get_config

router = APIRouter()
engine: Optional[InferenceEngine] = None


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    model_version: str
    class_index: int
    above_threshold: bool
    top_k: Optional[List[dict]] = None
    all_probabilities: Optional[dict] = None


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int


class ModelInfoResponse(BaseModel):
    model_version: str
    num_classes: int
    class_names: List[str]
    input_shape: List[int]
    output_shape: List[int]
    total_params: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


def get_engine() -> InferenceEngine:
    """Dependency to get the inference engine."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return engine


def _validate_options(top_k: Optional[int], confidence_threshold: Optional[float]) -> None:
    if top_k is not None and top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be >= 1")
    if confidence_threshold is not None and not 0 <= confidence_threshold <= 1:
        raise HTTPException(status_code=400, detail="confidence_threshold must be between 0 and 1")


@router.get('/health', response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status='healthy' if engine is not None else 'unhealthy',
        model_loaded=engine is not None,
        model_version=engine.model_version if engine else None,
    )


@router.get('/model', response_model=ModelInfoResponse)
async def get_model_info(eng: InferenceEngine = Depends(get_engine)):
    return ModelInfoResponse(**eng.get_model_info())


@router.post('/predict', response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    top_k: Optional[int] = Form(None),
    confidence_threshold: Optional[float] = Form(None),
    return_probabilities: bool = Form(True),
    eng: InferenceEngine = Depends(get_engine),
):
    """Predict a class without mutating shared engine state."""
    _validate_options(top_k, confidence_threshold)
    await _validate_file(file)
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Invalid image file: {exc}') from exc

    config = dict(eng.config)
    if top_k is not None:
        config['top_k'] = top_k
    if confidence_threshold is not None:
        config['confidence_threshold'] = confidence_threshold

    result = InferenceEngine(
        model=eng.model,
        class_names=eng.class_names,
        preprocessing=eng.preprocessing,
        config=config,
        model_version=eng.model_version,
        model_metadata=eng.model_metadata,
    ).predict(image, return_probabilities=return_probabilities)
    return PredictionResponse(**result)


@router.post('/predict/batch', response_model=BatchPredictionResponse)
async def predict_batch(
    files: List[UploadFile] = File(...),
    top_k: Optional[int] = Form(None),
    confidence_threshold: Optional[float] = Form(None),
    return_probabilities: bool = Form(True),
    eng: InferenceEngine = Depends(get_engine),
):
    """Predict multiple images without mutating shared engine state."""
    _validate_options(top_k, confidence_threshold)
    if not files:
        raise HTTPException(status_code=400, detail='At least one file is required')
    if len(files) > 100:
        raise HTTPException(status_code=400, detail='Maximum 100 files per batch')

    images = []
    for file in files:
        await _validate_file(file)
        contents = await file.read()
        try:
            images.append(Image.open(io.BytesIO(contents)).convert('RGB'))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f'Invalid image file: {exc}') from exc

    config = dict(eng.config)
    if top_k is not None:
        config['top_k'] = top_k
    if confidence_threshold is not None:
        config['confidence_threshold'] = confidence_threshold

    request_engine = InferenceEngine(
        model=eng.model,
        class_names=eng.class_names,
        preprocessing=eng.preprocessing,
        config=config,
        model_version=eng.model_version,
        model_metadata=eng.model_metadata,
    )
    results = request_engine.predict_batch(images, return_probabilities=return_probabilities)
    return BatchPredictionResponse(
        predictions=[PredictionResponse(**result) for result in results],
        total=len(results),
    )


async def _validate_file(file: UploadFile):
    """Validate upload extension, size, and image integrity."""
    api_config = get_config().api
    max_size = api_config.get('max_file_size', 10485760)
    allowed_extensions = api_config.get('allowed_extensions', ['.jpg', '.jpeg', '.png', '.bmp', '.webp'])

    filename = file.filename or ''
    if Path(filename).suffix.lower() not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f'Unsupported file format. Allowed: {allowed_extensions}')

    contents = await file.read()
    await file.seek(0)
    if len(contents) > max_size:
        raise HTTPException(status_code=413, detail=f'File too large. Maximum size: {max_size} bytes')
    try:
        Image.open(io.BytesIO(contents)).verify()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Invalid image file: {exc}') from exc


def initialize_engine(model_path: str, class_names: Optional[List[str]] = None) -> InferenceEngine:
    """Initialize the global inference engine."""
    global engine
    engine = create_inference_engine(model_path, class_names)
    return engine
