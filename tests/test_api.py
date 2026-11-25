import pytest


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "indexed_chunks": 3}


def test_ask_returns_answer_and_sources(client):
    r = client.post("/ask", json={"question": "What are the XP values?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Test answer"
    assert body["sources"] == [{"source": "a.pdf", "page": 3}]
    assert body["latency_ms"] >= 0


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "hi"},
        {"question": "x" * 1001},
        {"question": "valid question here", "mode": "banana"},
        {},
    ],
)
def test_ask_rejects_invalid_input(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_ask_hides_internal_errors(client, fake_chain):
    def boom(*args, **kwargs):
        raise RuntimeError("secret-key-sk-123")

    fake_chain.ask = boom
    r = client.post("/ask", json={"question": "valid question here"})
    assert r.status_code == 502
    assert "secret" not in r.text


def test_stream_sends_sources_then_tokens_then_done(client):
    r = client.post("/ask/stream", json={"question": "valid question here"})
    assert r.status_code == 200
    body = r.text
    assert body.index("event: sources") < body.index("event: token") < body.index("event: done")
    assert '"Hel"' in body
def test_ask_requires_key_when_configured(client, monkeypatch):
    from app import config

    monkeypatch.setattr(config, "API_KEY", "secret")
    q = {"question": "valid question here"}
    assert client.post("/ask", json=q).status_code == 401
    assert client.post("/ask", json=q, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/ask", json=q, headers={"X-API-Key": "secret"}).status_code == 200


def test_health_stays_open(client, monkeypatch):
    from app import config

    monkeypatch.setattr(config, "API_KEY", "secret")
    assert client.get("/health").status_code == 200