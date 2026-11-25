import importlib
import os
import sys
import types

import pytest

# Force test-safe settings BEFORE app.config is imported. load_dotenv() never
# overrides variables that are already set (even to an empty string), so a
# developer's local .env can't leak into the tests.
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["API_KEY"] = ""
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""


def _doc(source="a.pdf", page=2):
    return types.SimpleNamespace(
        page_content="text", metadata={"source": source, "page": page}
    )


@pytest.fixture
def fake_chain(monkeypatch):
    fake = types.ModuleType("app.chain")
    fake.retriever = types.SimpleNamespace(
        docs=[1, 2, 3], retrieve=lambda question, mode: [_doc()]
    )
    fake.ask = lambda question, mode="hybrid", effort=None: {
        "answer": "Test answer",
        "sources": [("a.pdf", 3)],
        "effort": "medium",
    }

    async def stream_answer(question, docs, effort=None):
        for token in ("Hel", "lo"):
            yield token

    fake.stream_answer = stream_answer

    import app

    monkeypatch.setitem(sys.modules, "app.chain", fake)
    monkeypatch.setattr(app, "chain", fake, raising=False)
    sys.modules.pop("app.api", None)
    yield fake
    sys.modules.pop("app.api", None)


@pytest.fixture
def client(fake_chain):
    from fastapi.testclient import TestClient

    api = importlib.import_module("app.api")
    return TestClient(api.app)