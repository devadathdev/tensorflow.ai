import json
import pytest

from src.deva.data import record_to_text, read_jsonl, validate_jsonl


def test_plain_text_record():
    assert record_to_text({"text": " hello "}) == "hello"


def test_messages_record_is_normalized():
    value = json.loads(record_to_text({
        "messages": [{"role": "user", "content": " hi "}]
    }))
    assert value["messages"][0]["content"] == "hi"


def test_invalid_message_is_rejected():
    with pytest.raises(ValueError):
        record_to_text({"messages": [{"role": "user", "content": ""}]})


def test_jsonl_validation(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"text":"one"}\n\n{"text":"two"}\n', encoding="utf-8")
    assert validate_jsonl(path)["records"] == 2
    assert len(list(read_jsonl(path))) == 2


def test_bad_jsonl_reports_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text":"ok"}\n{bad json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r":2:"):
        list(read_jsonl(path))
