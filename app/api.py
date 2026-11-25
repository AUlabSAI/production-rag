import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from langfuse import get_client
from pydantic import BaseModel, Field

from app import chain, config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rag")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

if not config.API_KEY:
    log.warning("API_KEY is not set: authentication is DISABLED")


def require_api_key(key: str | None = Depends(api_key_header)) -> None:
    if not config.API_KEY:
        return
    if not key or not secrets.compare_digest(key.encode(), config.API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    get_client().flush()


app = FastAPI(title="Production RAG API", version="0.1.0", lifespan=lifespan)
Mode = Literal["vector", "hybrid", "hybrid_rerank"]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: Mode = "hybrid_rerank"


class Source(BaseModel):
    source: str
    page: int


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: int


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": len(chain.retriever.docs)}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(req: AskRequest):
    # Plain `def`: FastAPI runs it in a worker thread, so the blocking
    # retrieval and LLM calls don't freeze the event loop.
    t0 = time.perf_counter()
    try:
        result = chain.ask(req.question, mode=req.mode)
    except Exception:
        log.exception("ask failed")
        raise HTTPException(status_code=502, detail="Upstream model error, try again.")
    latency_ms = int((time.perf_counter() - t0) * 1000)
    log.info("ask mode=%s latency_ms=%d q=%r", req.mode, latency_ms, req.question[:80])
    return AskResponse(
        answer=result["answer"],
        sources=[Source(source=s, page=p) for s, p in result["sources"]],
        latency_ms=latency_ms,
    )


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/ask/stream", dependencies=[Depends(require_api_key)])
async def ask_stream(req: AskRequest):
    async def events():
        try:
            docs = await run_in_threadpool(chain.retriever.retrieve, req.question, req.mode)
            sources = sorted({(d.metadata["source"], d.metadata["page"] + 1) for d in docs})
            yield sse("sources", [{"source": s, "page": p} for s, p in sources])
            async for token in chain.stream_answer(req.question, docs):
                yield sse("token", token)
            yield sse("done", {})
        except Exception:
            log.exception("stream failed")
            yield sse("error", "Upstream model error, try again.")

    return StreamingResponse(events(), media_type="text/event-stream")