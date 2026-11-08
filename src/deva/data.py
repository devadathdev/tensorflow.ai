"""JSONL instruction-data validation and TensorFlow-backed text streaming."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def record_to_text(record: dict) -> str:
    """Convert either a plain-text or chat-messages record into training text."""
    if not isinstance(record, dict):
        raise ValueError("Each JSONL record must be a JSON object.")
    text = record.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Record must contain non-empty 'text' or 'messages'.")
    rendered = []
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError("'messages' entries must be objects.")
        role, content = item.get("role"), item.get("content")
        if not isinstance(role, str) or not isinstance(content, str) or not content.strip():
            raise ValueError("Every message needs non-empty string role and content.")
        rendered.append({"role": role, "content": content.strip()})
    return json.dumps({"messages": rendered}, ensure_ascii=False)


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record_to_text(record)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            yield record


def validate_jsonl(path: str | Path) -> dict:
    count = 0
    for _ in read_jsonl(path):
        count += 1
    if count == 0:
        raise ValueError(f"Dataset is empty: {path}")
    return {"path": str(path), "records": count, "valid": True}


def tensorflow_text_batches(path: str | Path, batch_size: int = 32):
    """Stream raw JSONL lines with tf.data; TensorFlow is optional for data-only tests."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    import tensorflow as tf

    dataset = tf.data.TextLineDataset(str(path))
    dataset = dataset.filter(lambda line: tf.strings.length(tf.strings.strip(line)) > 0)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
