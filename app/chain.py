import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse import get_client, observe, propagate_attributes
from langfuse.langchain import CallbackHandler

from app import config
from app.retriever import HybridRetriever

retriever = HybridRetriever()  # built once at startup


def _make_llm(effort: str) -> ChatOpenAI:
    # generous timeout: high effort can think for a long time
    return ChatOpenAI(
        model=config.LLM_MODEL, reasoning_effort=effort, timeout=90, max_retries=2
    )


LLMS = {e: _make_llm(e) for e in ("low", "medium", "high")}

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You answer questions using ONLY the provided context. "
            "If the context does not contain the answer, say "
            "\"I don't have enough information in the documents to answer that.\" "
            "Do not use outside knowledge."
        ),
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

SYNTH = re.compile(
    r"\b(why|how (does|do|can|would)|distinguish\w*|differ\w*|compare\w*|relate\w*|explain)\b",
    re.IGNORECASE,
)
VALID_POLICIES = {"low", "medium", "high", "routed"}


def choose_effort(question: str) -> str:
    return config.ROUTE_HIGH if SYNTH.search(question) else config.ROUTE_LOW


def resolve_effort(question: str, policy: str | None = None) -> str:
    policy = policy or config.DEFAULT_EFFORT
    if policy not in VALID_POLICIES:
        raise ValueError(f"unknown effort policy: {policy!r}")
    return choose_effort(question) if policy == "routed" else policy


def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata['source']}, p.{d.metadata['page'] + 1}]\n{d.page_content}"
        for d in docs
    )


def _callbacks():
    return [CallbackHandler()] if config.TRACING_ENABLED else []


def generate(question: str, docs, callbacks=None, effort: str | None = None) -> str:
    e = resolve_effort(question, effort)
    return (prompt | LLMS[e]).invoke(
        {"context": format_docs(docs), "question": question},
        config={"callbacks": callbacks or []},
    ).content


@observe(name="retrieve", capture_output=False)
def traced_retrieve(question: str, mode: str):
    docs = retriever.retrieve(question, mode=mode)
    get_client().update_current_span(
        input={"question": question, "mode": mode},
        output=[
            {"source": d.metadata["source"], "page": d.metadata["page"] + 1,
             "preview": d.page_content[:200]}
            for d in docs
        ],
    )
    return docs


@observe(name="rag-ask")
def ask(question: str, mode: str = "hybrid_rerank", effort: str | None = None) -> dict:
    policy = effort or config.DEFAULT_EFFORT
    resolved = resolve_effort(question, policy)
    with propagate_attributes(
        tags=[f"mode:{mode}", f"policy:{policy}", f"effort:{resolved}"]
    ):
        docs = traced_retrieve(question, mode)
        answer = generate(question, docs, callbacks=_callbacks(), effort=resolved)
    sources = sorted({(d.metadata["source"], d.metadata["page"] + 1) for d in docs})
    return {"answer": answer, "sources": sources, "effort": resolved}


async def stream_answer(question: str, docs, effort: str | None = None):
    e = resolve_effort(question, effort)
    async for chunk in (prompt | LLMS[e]).astream(
        {"context": format_docs(docs), "question": question},
        config={"callbacks": _callbacks()},
    ):
        if chunk.content:
            yield chunk.content


if __name__ == "__main__":
    print(ask("What is Agile Methodology?"))