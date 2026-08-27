"""Models module for TensorVision AI."""

from src.models.cnn import CustomCNN
from src.models.transfer_learning import TransferLearningModel
from src.models.detection import ObjectDetectionModel
from src.models.segmentation import SegmentationModel
from src.models.ocr import OCRModel, OCRDetector
from src.models.factory import (
    create_model,
    create_detection_model,
    create_segmentation_model_fn as create_segmentation_model,
    create_ocr_model_fn as create_ocr_model,
    create_ocr_detector_fn as create_ocr_detector,
    compile_model,
    compile_detection_model,
    compile_segmentation_model,
    compile_ocr_model,
    get_model_info
)

__all__ = [
    'CustomCNN',
    'TransferLearningModel',
    'ObjectDetectionModel',
    'SegmentationModel',
    'OCRModel',
    'OCRDetector',
    'create_model',
    'create_detection_model',
    'create_segmentation_model',
    'create_ocr_model',
    'create_ocr_detector',
    'compile_model',
    'compile_detection_model',
    'compile_segmentation_model',
    'compile_ocr_model',
    'get_model_info',
]