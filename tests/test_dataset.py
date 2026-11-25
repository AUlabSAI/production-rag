import json
from pathlib import Path

import pytest

DATASET = Path("eval/dataset.jsonl")
REQUIRED = {"question", "reference_answer", "source", "pages", "answerable"}


@pytest.mark.skipif(not DATASET.exists(), reason="dataset not in repo")
def test_dataset_is_valid():
    lines = [l for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert lines
    for n, line in enumerate(lines, 1):
        item = json.loads(line)  # raises on bad JSON, e.g. a stray backslash
        assert REQUIRED <= item.keys(), f"line {n}: missing keys"
        if item["answerable"]:
            assert item["source"] and item["pages"], f"line {n}: needs source and pages"