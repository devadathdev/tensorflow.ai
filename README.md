# TensorVision AI

**End-to-end TensorFlow/Keras computer-vision platform** for classification, transfer learning, object detection, semantic segmentation, and OCR.

> Build → train → evaluate → export → serve vision models from one configurable project.

## Status

| Capability | Status |
|---|---|
| Image classification | ✅ Production pipeline |
| Transfer learning | ✅ Production pipeline |
| Semantic segmentation | ✅ Structured dataset + task-aware training |
| OCR recognition | ✅ CRNN + CTC structured pipeline |
| Object detection | 🟡 Functional baseline; SSD matching/decoding still needs hardening |
| FastAPI inference | ✅ Available |
| TensorFlow Lite | ✅ Available |
| Docker CPU/GPU | ✅ Available |
| Automated tests / CI | ✅ Configured |

## Why TensorVision AI?

TensorVision is designed around **explicit task contracts**. A detection, segmentation, or OCR job cannot silently fall back to a classification directory dataset. Each task declares its annotation format, dataset adapter, model, loss, and training path.

## Supported Tasks

### 1. Image Classification

Directory-based datasets are supported out of the box:

```text
data/
├── train/
│   ├── cats/
│   │   ├── cat001.jpg
│   │   └── cat002.jpg
│   └── dogs/
│       └── dog001.jpg
├── val/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```

Train a CNN:

```bash
python train.py --model-type cnn
```

Or use transfer learning:

```bash
python train.py --model-type transfer_learning --architecture efficientnetb0
```

### 2. Semantic Segmentation

Segmentation uses paired images and masks. Masks must contain integer class IDs in the range `0..num_classes-1`.

```yaml
dataset:
  segmentation_annotations:
    num_classes: 2
    train:
      images: data/segmentation/train/images
      masks: data/segmentation/train/masks
    val:
      images: data/segmentation/val/images
      masks: data/segmentation/val/masks
```

The task uses segmentation-specific targets, loss, and metrics rather than classification labels.

### 3. Object Detection

Detection accepts COCO-style JSON annotations:

```yaml
dataset:
  detection_annotations:
    train: data/detection/instances_train.json
    val: data/detection/instances_val.json
    image_root: data/detection
```

The adapter converts COCO bounding boxes into normalized fixed-size targets compatible with the existing SSD-style model interface.

**Important:** this is deliberately documented as a **functional baseline**, not a benchmark-ready SSD implementation. Proper production detection requires anchor generation, IoU matching, positive/negative assignment, hard-negative mining, target encoding/decoding, robust NMS, and mAP evaluation.

### 4. OCR

OCR recognition accepts JSON or JSONL annotations containing `image` and `text`:

```json
[
  {"image": "0001.jpg", "text": "hello world"},
  {"image": "0002.jpg", "text": "tensorvision"}
]
```

Configure:

```yaml
dataset:
  ocr_annotations:
    train: data/ocr/train.json
    val: data/ocr/val.json
    image_root: data/ocr
    max_length: 32
```

The recognizer uses CRNN + CTC. Character ID `0` is reserved for the CTC blank and character IDs begin at `1`. A DBNet-style text detector is also included for the OCR detection stage.

## Architecture

```text
                         TensorVision AI
                               │
                     ┌─────────┴─────────┐
                     │   Task Registry   │
                     └─────────┬─────────┘
                               │
        ┌──────────────┬───────┼────────┬──────────────┐
        ▼              ▼       ▼        ▼              ▼
  Classification  Transfer  Detection  Segmentation   OCR
        │          Learning     │          │           │
        ▼              ▼        ▼          ▼      ┌────┴────┐
   Directory       Pretrained  COCO      Masks   DBNet     CRNN
    Dataset         Backbone   Adapter    Adapter Detector Recognizer
        └──────────────┴────────┴──────────┴──────────────┘
                               │
                         Task-aware Trainer
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
                 Evaluate              Export
                    │                     │
                    ▼                     ▼
                 Metrics            Keras / TFLite
                                          │
                                          ▼
                                      FastAPI
```

## Project Structure

```text
tensorflow.ai/
├── data/                         # Local datasets
├── models/                       # Saved models
├── logs/                         # Training logs
├── src/
│   ├── api/                      # FastAPI routes and schemas
│   ├── data/
│   │   ├── dataset.py            # Classification + structured loaders
│   │   ├── preprocessing.py      # Image preprocessing
│   │   └── augmentation.py       # Training augmentation
│   ├── models/
│   │   ├── cnn.py                # Custom CNN
│   │   ├── transfer_learning.py  # Pretrained backbones
│   │   ├── detection.py          # SSD-style detector
│   │   ├── segmentation.py       # Segmentation models
│   │   ├── ocr.py                # DBNet + CRNN
│   │   └── factory.py            # Model creation + compilers
│   ├── training/
│   │   ├── trainer.py            # Task-aware trainer
│   │   └── callbacks.py          # Training callbacks
│   ├── evaluation/               # Metrics and evaluation
│   ├── inference/                # Inference engines
│   ├── tasks.py                  # Task/capability contracts
│   └── config.py                 # YAML + environment config
├── tests/                        # Unit / contract / integration tests
├── config.yaml
├── train.py
├── predict.py
├── predict_tflite.py
├── evaluate.py
├── serve.py
├── convert_tflite.py
├── Dockerfile
├── Dockerfile.gpu
├── docker-compose.yml
└── requirements.txt
```

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### Train classification

```bash
python train.py \
  --model-type transfer_learning \
  --architecture efficientnetb0 \
  --epochs 20 \
  --batch-size 32 \
  --lr 0.001
```

### Run inference

```bash
python predict.py path/to/image.jpg
```

```bash
python predict.py image.jpg --top-k 5 --threshold 0.3 --output results.json
```

### Start the API

```bash
python serve.py
```

Routes:

```text
GET  /api/v1/health
GET  /api/v1/model
POST /api/v1/predict
POST /api/v1/predict/batch
```

### Export to TensorFlow Lite

```bash
python convert_tflite.py
python predict_tflite.py path/to/image.jpg
```

## Configuration

Example:

```yaml
dataset:
  train_dir: data/train
  val_dir: data/val
  test_dir: data/test
  image_size: [224, 224]
  batch_size: 32

model:
  type: transfer_learning
  architecture: efficientnetb0

training:
  epochs: 50
  learning_rate: 0.001
  optimizer: adam

augmentation:
  enabled: true
  horizontal_flip: true
  rotation_factor: 0.1
```

Environment overrides include `TENSORVISION_EPOCHS`, `TENSORVISION_BATCH_SIZE`, `TENSORVISION_LEARNING_RATE`, `TENSORVISION_MODEL_DIR`, `TENSORVISION_USE_GPU`, and `TENSORVISION_API_PORT`.

## Model Backbones

### Classification / Transfer Learning

- MobileNetV2
- MobileNetV3 Small / Large
- EfficientNet B0–B3
- EfficientNetV2 B0–B3
- ResNet50 / 101 / 152
- ResNet50V2
- InceptionV3
- Xception
- DenseNet121 / 169 / 201

### Detection

- MobileNetV2
- MobileNetV3 Small / Large
- EfficientNet B0–B3
- ResNet50 / 101
- ResNet50V2

### OCR

- MobileNetV2
- MobileNetV3 Small / Large
- EfficientNet B0 / B1
- Custom ResNet34

## Training Design

| Task | Dataset | Training objective |
|---|---|---|
| CNN | Directory labels | Sparse categorical classification |
| Transfer learning | Directory labels | Sparse categorical classification |
| Detection | COCO JSON | Box regression + focal classification baseline |
| Segmentation | Image/mask pairs | Cross-entropy + Dice |
| OCR | JSON/JSONL | CTC sequence recognition |

Classification class weights are applied only to classification tasks. Structured tasks receive structured targets and task-specific compilation.

## Testing

```bash
pytest tests/ -q
pytest tests/ --cov=src --cov-report=html
```

The test suite includes task contracts and structured dataset adapters. Full model tests may require TensorFlow-compatible resources and pretrained-weight downloads.

## Deployment

### Docker CPU

```bash
docker build -t tensorvision-ai .
docker run --rm -p 8000:8000 -v $(pwd)/models:/app/models tensorvision-ai
```

### Docker Compose

```bash
docker compose up --build
```

GPU deployment is available through the GPU Docker configuration when the host has a compatible NVIDIA/TensorFlow environment.

## Engineering Notes

- Configuration instances are independent and do not silently reuse another YAML configuration.
- Structured tasks fail fast when required annotation configuration is missing.
- OCR labels reserve `0` for CTC blank.
- Segmentation masks use nearest-neighbor resizing to preserve class IDs.
- Detection currently uses fixed target slots and should not be represented as a final SSD implementation.
- The project intentionally documents incomplete areas instead of claiming unsupported functionality.

## Roadmap

- [x] Core classification pipeline
- [x] Transfer learning
- [x] Data augmentation
- [x] Training callbacks and monitoring
- [x] FastAPI deployment
- [x] TensorFlow Lite export
- [x] Structured segmentation pipeline
- [x] OCR CRNN/CTC pipeline
- [x] COCO detection adapter
- [x] Task-aware training
- [x] Automated CI
- [ ] Production SSD anchor generator + IoU matcher
- [ ] Detection box decoder + robust NMS evaluation
- [ ] COCO mAP evaluation
- [ ] DBNet polygon/quad extraction and end-to-end OCR
- [ ] Experiment tracking and model registry
- [ ] Distributed multi-GPU training
- [ ] Streaming/video inference

## License

MIT License
