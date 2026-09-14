# TensorVision AI

End-to-end TensorFlow/Keras vision platform for classification, object detection, semantic segmentation, and OCR.

## Production task pipeline

TensorVision now selects a task-specific dataset and compiler instead of silently feeding classification labels into every model. Classification keeps the directory contract; structured tasks use explicit annotation contracts.

### Classification

```text
data/train/<class>/*.jpg
data/val/<class>/*.jpg
data/test/<class>/*.jpg
```

Run:

```bash
python train.py --model-type cnn
python train.py --model-type transfer_learning --architecture efficientnetb0
```

### Segmentation

Configure `dataset.segmentation_annotations` with matching image/mask directories. Masks must contain integer class IDs from `0..num_classes-1`.

```yaml
dataset:
  segmentation_annotations:
    num_classes: 2
    train: {images: data/segmentation/train/images, masks: data/segmentation/train/masks}
    val: {images: data/segmentation/val/images, masks: data/segmentation/val/masks}
```

### Detection

Configure COCO JSON annotations:

```yaml
dataset:
  detection_annotations:
    train: data/detection/instances_train.json
    val: data/detection/instances_val.json
    image_root: data/detection
```

The adapter normalizes COCO boxes into the current SSD head's fixed prediction slots. **Important:** this is a functional baseline adapter, not a full SSD anchor encoder/matcher. For production-grade detection accuracy, anchor generation, IoU matching, hard-negative mining, and box decoding should be added before benchmarking the detector.

### OCR

Use JSON or JSONL records with `image` and `text` fields:

```json
[{"image":"0001.jpg","text":"hello"}]
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

The CRNN uses CTC loss; `0` is reserved for the CTC blank and character IDs start at `1`.

## Features

- Custom CNN and transfer learning
- MobileNet, EfficientNet, ResNet and other pretrained backbones
- Structured detection, segmentation and OCR dataset adapters
- Task-aware compilation and training callbacks
- Dataset validation and class weighting for classification
- TensorBoard, early stopping, LR scheduling and checkpointing
- Versioned model output and metadata
- FastAPI inference service
- TensorFlow Lite conversion
- Docker CPU/GPU deployment
- Automated pytest CI on pushes and pull requests

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Testing

```bash
pytest tests/ -q
pytest tests/ --cov=src --cov-report=html
```

The repository includes contract tests for task selection and structured segmentation/OCR/COCO loaders. CI runs the full test suite; TensorFlow model downloads/training can require network access and additional compute.

## Project layout

```text
tensorflow.ai/
├── data/
├── models/
├── logs/
├── src/
│   ├── api/
│   ├── data/          # classification + structured adapters
│   ├── evaluation/
│   ├── inference/
│   ├── models/        # CNN, TL, detection, segmentation, OCR
│   ├── training/
│   └── tasks.py
├── tests/
├── config.yaml
├── train.py
├── predict.py
├── evaluate.py
├── serve.py
└── convert_tflite.py
```

## API

```bash
python serve.py
```

Endpoints:
- `GET /api/v1/health`
- `GET /api/v1/model`
- `POST /api/v1/predict`
- `POST /api/v1/predict/batch`

## License

MIT License
