import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app import config
from app.ingest import get_vectorstore


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rrf(rank_lists: list[list[Document]], k: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion: merge ranked lists using rank positions only."""
    fused = {}
    for ranked in rank_lists:
        for rank, doc in enumerate(ranked):
            entry = fused.setdefault(doc.page_content, [0.0, doc])
            entry[0] += 1 / (k + rank + 1)
    return [doc for _, doc in sorted(fused.values(), key=lambda e: -e[0])]


class HybridRetriever:
    def __init__(self):
        self.vs = get_vectorstore()
        data = self.vs.get(include=["documents", "metadatas"])
        self.docs = [
            Document(page_content=t, metadata=m)
            for t, m in zip(data["documents"], data["metadatas"])
        ]
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in self.docs])
        self.reranker = CrossEncoder(config.RERANK_MODEL)

    def _vector(self, query: str) -> list[Document]:
        return self.vs.similarity_search(query, k=config.CANDIDATES)

    def _bm25(self, query: str) -> list[Document]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[: config.CANDIDATES]
        return [self.docs[i] for i in top]

    def _rerank(self, query: str, docs: list[Document]) -> list[Document]:
        scores = self.reranker.predict([(query, d.page_content) for d in docs])
        ranked = sorted(zip(scores, docs), key=lambda p: -p[0])
        return [d for _, d in ranked]

    def retrieve(self, query: str, mode: str = "hybrid_rerank") -> list[Document]:
        if mode == "vector":
            return self._vector(query)[: config.TOP_K]
        fused = rrf([self._vector(query), self._bm25(query)])
        if mode == "hybrid":
            return fused[: config.TOP_K]
        return self._rerank(query, fused[: config.CANDIDATES])[: config.TOP_K]