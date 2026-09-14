"""Task definitions and validation for TensorVision AI."""

SUPPORTED_TASKS = {
    'cnn': 'image classification with a custom CNN',
    'transfer_learning': 'image classification with a pretrained backbone',
    'detection': 'object detection with bounding-box annotations',
    'segmentation': 'semantic segmentation with pixel masks',
    'ocr': 'text detection/recognition with OCR annotations',
}

CLASSIFICATION_TASKS = {'cnn', 'transfer_learning'}
ANNOTATION_KEYS = {
    'detection': 'dataset.detection_annotations',
    'segmentation': 'dataset.segmentation_annotations',
    'ocr': 'dataset.ocr_annotations',
}


def validate_task(task: str) -> None:
    """Raise a clear error when an unsupported task is selected."""
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported task '{task}'. Available tasks: {', '.join(SUPPORTED_TASKS)}"
        )


def requires_annotations(task: str) -> bool:
    """Return whether a task needs structured annotations."""
    validate_task(task)
    return task not in CLASSIFICATION_TASKS


def annotation_config_key(task: str) -> str:
    """Return the config key used for structured annotations."""
    validate_task(task)
    if task in CLASSIFICATION_TASKS:
        raise ValueError(f"Task '{task}' uses directory labels and has no annotation key")
    return ANNOTATION_KEYS[task]
