import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app import config
from app.chain import format_docs, generate, resolve_effort, retriever

TAG = sys.argv[1] if len(sys.argv) > 1 else "default"
MODES = os.getenv("MODES", "vector,hybrid,hybrid_rerank").split(",")


class Verdict(BaseModel):
    faithful: bool
    correct: bool


# Judge is pinned so it never changes between experiments
judge = ChatOpenAI(model=config.LLM_MODEL, reasoning_effort="medium").with_structured_output(Verdict)

JUDGE_PROMPT = """You are grading a RAG system's answer.

Question: {question}
Reference answer: {reference}
Retrieved context:
{context}
System answer: {answer}

Return two booleans:
- faithful: true if every factual claim in the system answer is supported by the
  retrieved context. A refusal such as "I don't have enough information" is faithful.
- correct: true if the system answer contains the key facts of the reference answer
  without contradicting it. If the reference answer is "NOT_ANSWERABLE", correct is
  true only if the system answer says it does not have enough information."""


def load(path="eval/dataset.jsonl"):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def first_hit_rank(docs, item):
    for i, d in enumerate(docs, 1):
        if d.metadata["source"] == item["source"] and d.metadata["page"] + 1 in item["pages"]:
            return i
    return None


def run(mode, data):
    rows = []
    for n, item in enumerate(data, 1):
        t0 = time.perf_counter()
        docs = retriever.retrieve(item["question"], mode=mode)
        retrieval_s = time.perf_counter() - t0

        effort_used = resolve_effort(item["question"], config.DEFAULT_EFFORT)
        t1 = time.perf_counter()
        answer = generate(item["question"], docs, effort=effort_used)
        generation_s = time.perf_counter() - t1

        reference = item["reference_answer"] if item["answerable"] else "NOT_ANSWERABLE"
        verdict = judge.invoke(JUDGE_PROMPT.format(
            question=item["question"], reference=reference,
            context=format_docs(docs), answer=answer,
        ))
        rows.append({
            "question": item["question"],
            "type": item.get("type", "unknown"),
            "answerable": item["answerable"],
            "rank": first_hit_rank(docs, item) if item["answerable"] else None,
            "faithful": verdict.faithful,
            "correct": verdict.correct,
            "effort_used": effort_used,
            "retrieval_s": retrieval_s,
            "generation_s": generation_s,
            "answer": answer,
            "retrieved": [(d.metadata["source"], d.metadata["page"] + 1) for d in docs],
        })
        print(f"  [{mode}] {n}/{len(data)}", end="\r")
    print()
    return rows


def summarize(rows):
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    return {
        "hit@1": mean(r["rank"] == 1 for r in ans) if ans else 0,
        f"hit@{config.TOP_K}": mean(r["rank"] is not None for r in ans) if ans else 0,
        "MRR": mean(1 / r["rank"] if r["rank"] else 0 for r in ans) if ans else 0,
        "correct": mean(r["correct"] for r in ans) if ans else 0,
        "faithful": mean(r["faithful"] for r in rows),
        "refusal_acc": mean(r["correct"] for r in unans) if unans else float("nan"),
        "retrieval_s": mean(r["retrieval_s"] for r in rows),
        "generation_s": mean(r["generation_s"] for r in rows),
    }


if __name__ == "__main__":
    data = load()
    print(f"{len(data)} questions ({sum(d['answerable'] for d in data)} answerable), "
          f"policy={config.DEFAULT_EFFORT}, tag={TAG}")
    results, summaries = {}, {}
    for mode in MODES:
        results[mode] = run(mode, data)
        summaries[mode] = summarize(results[mode])

    Path(f"eval/results_{TAG}.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    cols = list(next(iter(summaries.values())).keys())
    print("\n| mode | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for mode, s in summaries.items():
        print(f"| {mode} | " + " | ".join(f"{s[c]:.2f}" for c in cols) + " |")