from langchain_core.documents import Document

from app.retriever import rrf, tokenize


def doc(text):
    return Document(page_content=text, metadata={})


def test_tokenize_lowercases_and_splits_punctuation():
    assert tokenize("O(log n), ELSIF!") == ["o", "log", "n", "elsif"]


def test_rrf_ranks_docs_found_by_both_lists_first():
    a, b, c = doc("a"), doc("b"), doc("c")
    fused = rrf([[a, b], [c, b]])
    assert fused[0].page_content == "b"
    assert {d.page_content for d in fused} == {"a", "b", "c"}


def test_rrf_handles_empty_lists():
    assert rrf([[], []]) == []