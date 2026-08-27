# TensorVision AI

End-to-end image classification platform built with TensorFlow/Keras. Transform labeled image datasets into trained, evaluated, and deployable AI models.

## Features

- **Dataset Management**: Automatic class discovery, validation, and statistics
- **Data Pipeline**: Optimized TensorFlow data loading with preprocessing, batching, shuffling, prefetching
- **Data Augmentation**: Configurable transformations (flip, rotate, zoom, brightness, contrast)
- **Model Training**: Custom CNN and transfer learning (MobileNet, EfficientNet, ResNet, etc.)
- **Training Monitoring**: TensorBoard integration, early stopping, learning rate scheduling
- **Model Evaluation**: Accuracy, precision, recall, F1, confusion matrix, per-class metrics, ROC AUC
- **Model Management**: Versioned model storage with metadata
- **Inference Engine**: Single and batch prediction with confidence scores
- **REST API**: FastAPI service with `/predict`, `/predict/batch`, `/health`, `/model` endpoints

## Architecture

```
tensorflow-ai/
├── data/                 # Dataset directories (train/val/test)
├── models/               # Trained model checkpoints
├── logs/                 # TensorBoard and CSV logs
├── src/
│   ├── config.py         # Configuration management
│   ├── data/             # Dataset, preprocessing, augmentation
│   ├── models/           # CNN, transfer learning, factory
│   ├── training/         # Trainer, callbacks
│   ├── evaluation/       # Metrics, evaluator
│   ├── inference/        # Engine, predictor
│   └── api/              # FastAPI server
├── tests/                # Unit and integration tests
├── config.yaml           # Configuration
├── requirements.txt      # Dependencies
├── train.py              # Training CLI
├── predict.py            # Inference CLI
├── evaluate.py           # Evaluation CLI
└── serve.py              # API server CLI
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Dataset

Organize images in the following structure:

```
data/
├── train/
│   ├── class_a/
│   │   ├── img1.jpg
│   │   └── ...
│   └── class_b/
│       ├── img1.jpg
│       └── ...
├── val/
│   ├── class_a/
│   └── class_b/
└── test/  (optional)
    ├── class_a/
    └── class_b/
```

### 2. Train Model

```bash
# Basic training
python train.py

# With custom parameters
python train.py --epochs 100 --batch-size 64 --lr 0.001

# Transfer learning
python train.py --model-type transfer_learning --architecture efficientnetb0
```

### 3. Run Inference

```bash
# Single image
python predict.py path/to/image.jpg

# Directory of images
python predict.py path/to/images/

# With options
python predict.py image.jpg --top-k 3 --threshold 0.3 --output results.json

# Benchmark
python predict.py image.jpg --benchmark --runs 100
```

### 4. Start API Server

```bash
python serve.py

# Custom host/port
python serve.py --host 0.0.0.0 --port 8080
```

API endpoints:
- `GET /api/v1/health` - Health check
- `GET /api/v1/model` - Model information
- `POST /api/v1/predict` - Single image prediction
- `POST /api/v1/predict/batch` - Batch prediction

### 5. Evaluate Model

```bash
# On validation set
python evaluate.py

# On test set
python evaluate.py --split test

# Save results
python evaluate.py --output eval_results.json
```

## Configuration

All parameters configurable via `config.yaml`:

```yaml
dataset:
  train_dir: "data/train"
  val_dir: "data/val"
  image_size: [224, 224]
  batch_size: 32

model:
  type: "transfer_learning"  # or "cnn"
  architecture: "efficientnetb0"

training:
  epochs: 50
  learning_rate: 1e-3
  optimizer: "adam"

augmentation:
  enabled: true
  horizontal_flip: true
  rotation_factor: 0.1
```

Environment variable overrides:
- `TENSORVISION_EPOCHS`
- `TENSORVISION_BATCH_SIZE`
- `TENSORVISION_LEARNING_RATE`
- `TENSORVISION_MODEL_DIR`
- `TENSORVISION_USE_GPU`

## Model Architectures

### Custom CNN
Configurable convolutional blocks with batch normalization and dropout.

### Transfer Learning
Supported backbones:
- MobileNetV2, MobileNetV3 (Small/Large)
- EfficientNet B0-B3, EfficientNetV2 B0-B3
- ResNet50, ResNet101, ResNet152, ResNet50V2
- InceptionV3, Xception
- DenseNet121, DenseNet169, DenseNet201

## Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "serve.py"]
```

```bash
docker build -t tensorvision .
docker run -p 8000:8000 -v $(pwd)/models:/app/models tensorvision
```

## Roadmap

- [ ] Phase 1: MVP (Core pipeline) ✓
- [ ] Phase 2: Advanced ML (Augmentation, transfer learning, TensorBoard)
- [ ] Phase 3: Deployment (API, Docker, TensorFlow Lite)
- [ ] Phase 4: Platform Expansion (Detection, segmentation, OCR)

## License

MIT License