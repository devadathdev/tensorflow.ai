"""FastAPI server for TensorVision AI."""

import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import get_config
from src.api.routes import router, initialize_engine
from src.inference.engine import create_inference_engine


app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    config = get_config()
    api_config = config.api
    output_config = config.output
    
    model_dir = Path(output_config.get('model_dir', 'models'))
    model_name = output_config.get('model_name', 'tensorvision_model')
    
    model_files = list(model_dir.glob(f"{model_name}_best.keras"))
    if not model_files:
        model_files = list(model_dir.glob("*.keras"))
    if not model_files:
        model_files = list(model_dir.glob("*.h5"))
    if not model_files:
        model_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        if model_dirs:
            model_files = model_dirs
    
    if model_files:
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        try:
            eng = create_inference_engine(str(latest_model))
            app_state['engine'] = eng
            print(f"Loaded model: {latest_model}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    else:
        print("No model found. API will return 503 until model is loaded.")
    
    yield
    
    app_state.clear()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    config = get_config()
    api_config = config.api
    
    app = FastAPI(
        title="TensorVision AI API",
        description="Image Classification Inference API",
        version="1.0.0",
        lifespan=lifespan
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.include_router(router, prefix="/api/v1")
    
    @app.get("/")
    async def root():
        return {
            "name": "TensorVision AI API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health"
        }
    
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False
):
    """Run the API server."""
    import uvicorn
    uvicorn.run(
        "src.api.server:create_app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        factory=True
    )


if __name__ == "__main__":
    config = get_config()
    api_config = config.api
    run_server(
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000),
        workers=api_config.get('workers', 1)
    )