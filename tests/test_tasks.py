"""Tests for task capability and API safety contracts."""

import pytest

from src.tasks import SUPPORTED_TASKS, annotation_config_key, requires_annotations, validate_task


def test_supported_tasks_are_explicit():
    assert set(SUPPORTED_TASKS) == {'cnn', 'transfer_learning', 'detection', 'segmentation', 'ocr'}


def test_classification_does_not_require_annotations():
    assert requires_annotations('cnn') is False
    assert requires_annotations('transfer_learning') is False


def test_structured_tasks_require_annotations():
    assert requires_annotations('detection') is True
    assert requires_annotations('segmentation') is True
    assert requires_annotations('ocr') is True


def test_annotation_keys():
    assert annotation_config_key('detection') == 'dataset.detection_annotations'
    assert annotation_config_key('segmentation') == 'dataset.segmentation_annotations'
    assert annotation_config_key('ocr') == 'dataset.ocr_annotations'


def test_unknown_task_fails_clearly():
    with pytest.raises(ValueError, match='Unsupported task'):
        validate_task('video_generation')
