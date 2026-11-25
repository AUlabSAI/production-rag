# Production RAG over Course Notes

![CI](https://github.com/AUlabSAI/production-rag/actions/workflows/ci.yml/badge.svg)

A retrieval-augmented question-answering service over a set of PDF course notes. It answers only from the documents, cites the source file and page for every answer, and refuses when the notes don't contain the answer. The project covers the full path from ingestion to a deployed, tested, traced and evaluated API.

**Stack:** FastAPI, Chroma, BM25 + vector hybrid retrieval (optional cross-encoder reranker), OpenAI `gpt-5-mini` and `text-embedding-3-small`, Langfuse tracing, Docker, `uv`, pytest, ruff, GitHub Actions.

## Architecture

```mermaid
flowchart LR
  P[PDF course notes] --> I[ingest: chunk + embed]
  I --> C[(Chroma index)]
  C -->|built at startup| B[In-memory BM25 index]
  U[Client] -->|X-API-Key| A[FastAPI]
  A --> R[Retriever]
  R -->|vector search| C
  R -->|keyword search| B
  R -.optional.-> X[Cross-encoder reranker]
  A --> L[gpt-5-mini]
  A -.traces.-> F[Langfuse]
```

**Retrieval modes** (selectable per request with `mode`):

| Mode | What it does |
|---|---|
| `vector` | Embedding similarity search in Chroma |
| `hybrid` (default) | Vector + BM25 results merged with Reciprocal Rank Fusion |
| `hybrid_rerank` | Hybrid candidates re-scored by a cross-encoder (`ms-marco-MiniLM-L-6-v2`), loaded lazily on first use |

**Answering:** the prompt restricts the model to the retrieved context and tells it to say "I don't have enough information in the documents to answer that." when the context doesn't contain the answer. Each answer comes back with `source` and `page` citations.

## Results

Evaluated on **36 questions** (30 answerable, 6 unanswerable) written by hand from the course notes, using `gpt-5-mini`. Four runs were made, one per reasoning-effort setting (high, low, medium, and a routed policy). Read the numbers as indicative rather than precise: with 30 answerable questions, one question is worth 3.3 points, and no configuration was repeated, so run-to-run noise could not be measured. These runs used an earlier version of the evaluation harness, which did not record the effort setting in the result files.

**1. Baseline (reasoning effort = high)**

| mode | hit@4 | MRR | correct | faithful | refusal_acc | retrieval (s) |
|---|---|---|---|---|---|---|
| vector | 1.00 | 0.98 | 0.97 | 0.97 | 0.83 | 0.50 |
| hybrid | 1.00 | 0.97 | 0.90 | 0.97 | 1.00 | 0.46 |
| hybrid_rerank | 1.00 | 0.96 | 0.90 | 0.94 | 1.00 | 1.33 |

**2. Reasoning effort = low**

| mode | hit@4 | MRR | correct | faithful | refusal_acc | retrieval (s) |
|---|---|---|---|---|---|---|
| vector | 1.00 | 0.98 | 0.83 | 0.86 | 1.00 | 0.49 |
| hybrid | 1.00 | 0.97 | 0.73 | 0.81 | 1.00 | 0.43 |
| hybrid_rerank | 1.00 | 0.96 | 0.80 | 0.83 | 1.00 | 1.19 |

**3. Reasoning effort = medium**

| mode | hit@4 | MRR | correct | faithful | refusal_acc | retrieval (s) |
|---|---|---|---|---|---|---|
| vector | 1.00 | 0.98 | 0.97 | 0.97 | 1.00 | 0.51 |
| hybrid | 1.00 | 0.97 | 0.97 | 0.97 | 1.00 | 0.46 |
| hybrid_rerank | 1.00 | 0.96 | 0.93 | 0.94 | 1.00 | 1.24 |

**4. Effort = routed**

| mode | hit@4 | MRR | correct | faithful | refusal_acc | retrieval (s) |
|---|---|---|---|---|---|---|
| vector | 1.00 | 0.98 | 0.97 | 0.92 | 1.00 | 0.50 |
| hybrid | 1.00 | 0.97 | 1.00 | 1.00 | 1.00 | 0.44 |
| hybrid_rerank | 1.00 | 0.96 | 0.87 | 0.92 | 1.00 | 1.25 |

`routed` is the keyword-based effort policy: explanatory "why/how" questions run at the `ROUTE_HIGH` tier and all other questions at the `ROUTE_LOW` tier (defaults: high and low).

**What the evaluation shows**

- **Retrieval is saturated on this corpus.** `hit@4` is 1.00 for every mode in every run, and MRR is the same in all four runs (vector 0.98, hybrid 0.97, hybrid_rerank 0.96), because retrieval doesn't depend on the LLM setting. The right page lands in the top four chunks however you search, so the answer-quality columns are where the runs differ.
- **Vector and hybrid have no consistent winner.** `correct` for vector was 0.97, 0.83, 0.97, 0.97 across the four runs and for hybrid 0.90, 0.73, 0.97, 1.00. Hybrid was below vector at high effort and above it when routed. With no repeated runs of the same configuration, differences of a question or two can't be told apart from noise.
- **The reranker never scored best and was the slowest.** At high, medium and routed effort it had the lowest or tied-lowest `correct` score (0.90, 0.93, 0.87). Pooled over those three runs, `correct` was 96.7% for vector, 95.6% for hybrid and 90.0% for the reranker (the runs reuse the same 30 questions, so these are not independent trials). That points to no benefit, and possibly a small cost, on this corpus; I have not inspected the failing cases, so I don't claim a cause. Retrieval with the reranker took about 0.8 s longer than hybrid in every run (roughly 2.8x: 1.19-1.33 s vs 0.43-0.46 s), so it is optional and loaded only when requested.
- **`hybrid` is the default for a design reason, not a measured win:** BM25 handles exact terms (names, acronyms, codes) that embeddings can blur, and its retrieval latency was no higher than vector's in any of the four runs (0.43-0.46 s vs 0.49-0.51 s; vector's includes an embedding API call).
- **Low effort hurt quality; high effort did not help.** At low effort, `correct` was 0.73-0.83 in all three modes; in the other three runs the lowest score was 0.87, so the ranges don't overlap. Averaged over the three modes, `correct` was 0.92 at high effort, 0.96 at medium, 0.94 routed and 0.79 at low. `gpt-5-mini` spends hidden reasoning tokens before answering, so lower effort cuts them, and on this task that cost accuracy. The low-effort result is one run on 30 questions, so treat it as directional. The service uses medium effort.
- **Routing matched fixed-effort quality in one run.** With the default tiers, questions that don't look explanatory ran at low effort, yet `correct` averaged 0.94, in line with fixed medium (0.96) and fixed high (0.92). That is encouraging, but it is a single run, and per-question effort and latency were not recorded for it, so I don't claim a speed or cost saving. Medium stays the default until routing is measured against it with the current harness.
- **Refusals were reliable.** Every mode refused every unanswerable question in every run, except vector at high effort, which answered one of the six (`refusal_acc` 0.83).

**Metric definitions**

- `hit@4`: the labeled page (source file + page) appears in the top 4 retrieved chunks. Labels are page-level so they stay valid if chunking changes.
- `MRR`: mean reciprocal rank of the first correct page.
- `correct`: an LLM judge checks that the answer contains the key facts of the reference answer. For unanswerable questions, correct means the system refused.
- `faithful`: an LLM judge checks that every claim is supported by the retrieved context.
- `refusal_acc`: share of unanswerable questions the system correctly refused.
- The judge uses the same model family as the generator, which can make it lenient. In the current code the judge is pinned to medium reasoning effort, separate from the generator's setting.

**Live service (Langfuse traces, 13 requests, Docker container, medium effort)**

| Metric | Value |
|---|---|
| End-to-end latency | mean 6.8 s, median 6.2 s, p90 9.4 s |
| Share of time in the LLM call | about 78% |
| Retrieval latency | `hybrid` about 0.7 s, `vector` about 0.9 s (median 0.5 s), `hybrid_rerank` about 2.3 s |
| Tokens per request | about 890 input, 60 visible output, 210 hidden reasoning |
| Cost | about $0.0008 per request (about $0.77 per 1,000) |
| Docker image | 4.72 GB on disk (about 1 GB compressed) |

This is a small sample, and the reranker figure includes a lazy model load on at least one request, so treat the latencies as rough.

## Quickstart (Docker)

You need Docker with the compose plugin, an OpenAI API key, and your own PDFs. The course notes used for the evaluation are private and not included in this repo.

```bash
git clone https://github.com/AUlabSAI/production-rag.git
cd production-rag

cp .env.example .env          # then fill in the values (see Configuration)
# put your PDFs in data/pdfs/

docker compose build
docker compose run --rm ingest    # builds the index (run once, and after adding PDFs)
docker compose up -d api
```

After adding or changing PDFs, run `ingest` again and then `docker compose restart api`, because the BM25 index is built at startup.

Open `http://localhost:8000/docs` for the interactive API (use **Authorize** to enter your `API_KEY`).

### API

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"question": "What are the five Extreme Programming values?", "mode": "hybrid"}'
```

Response shape (illustrative):

```json
{
  "answer": "...",
  "sources": [{"source": "Agile Methodology.pdf", "page": 3}],
  "latency_ms": 5400
}
```

| Endpoint | Description |
|---|---|
| `POST /ask` | Full answer with citations |
| `POST /ask/stream` | Server-sent events: a `sources` event first, then `token` events, then `done` (or `error`) |
| `GET /health` | Liveness check with the number of indexed chunks (no auth required) |

Questions must be 3-1,000 characters, and `mode` must be one of the three retrieval modes. Invalid input returns 422, and internal failures return a generic 502 without exposing details.

## Configuration

Set in `.env`:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required |
| `API_KEY` | Clients must send it in the `X-API-Key` header. If empty, authentication is disabled and a warning is logged |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | Optional. Enables tracing when both keys are set |
| `REASONING_EFFORT` | `low`, `medium` (default), `high` or `routed` |

Defaults such as chunk size (1000 / overlap 100), `TOP_K` (4), the models, and the reranker model live in `app/config.py`. An experimental `routed` effort policy (a keyword router that sends explanatory "why/how" questions to the `ROUTE_HIGH` tier and everything else to `ROUTE_LOW`; defaults high and low) is off by default; see the results above for its single evaluation run.

## Development

```bash
uv sync
uv run pytest -q          # 14 tests, no network or API keys needed
uv run ruff check .
uv run python -m app.ingest
uv run uvicorn app.api:app --port 8000
```

CI (GitHub Actions) runs lint and tests, then builds the Docker image, on every push to `main` and on pull requests.

To run the evaluation you need your own documents and a matching `eval/dataset.jsonl`:

```bash
uv run python -m eval.validate                  # checks the dataset format
MODES=vector,hybrid uv run python -m eval.run_eval my_run   # writes eval/results_my_run.json
uv run python -m eval.show_failures eval/results_my_run.json hybrid
```

On Windows PowerShell, set the variable first: `$env:MODES = "vector,hybrid"`.

## Project layout

```
app/
  config.py      settings and constants
  ingest.py      PDF loading, chunking, idempotent indexing into Chroma
  retriever.py   vector, BM25, RRF fusion, lazy reranker
  chain.py       prompt, generation, reasoning-effort policy, Langfuse tracing
  api.py         FastAPI app: auth, validation, streaming
eval/            dataset, evaluation runner, failure inspection
tests/           API contract, retrieval fusion and dataset tests
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## Design decisions

- **Reciprocal Rank Fusion, implemented by hand.** RRF merges the vector and BM25 lists using rank positions only, so scores from the two systems never need to be normalized. It is about 15 lines and easy to reason about.
- **Idempotent ingestion.** Chunk IDs are content hashes, so re-running ingest updates the index instead of duplicating chunks.
- **The index lives in a Docker volume, not the image.** Building it during `docker build` would put the OpenAI key into the build, and a volume survives image rebuilds. The reranker model gets its own cache volume.
- **Lazy reranker.** The default mode doesn't need it, so PyTorch and the model load on the first `hybrid_rerank` request rather than at startup.
- **Evaluation measures retrieval and generation separately,** so a bad answer can be traced to either a retrieval miss or a generation problem.
- **Streaming sends sources before tokens,** so a client can show citations while the answer is still being written.
- **Authentication with a constant-time comparison** and a health endpoint left open for container health checks.
- **Tests use a fake `app.chain`,** so the suite is fast and free, and cannot be affected by a developer's local `.env`.

## Limitations and next steps

- **PDF tables are extracted poorly.** The loader reads text in layout order, so table cells can come out interleaved (seen on a comparison table in the notes). Questions that depend on matching cells to columns may be unreliable. Next step: a layout-aware parser, measured against the existing evaluation.
- **Small evaluation.** 36 questions from one document set, one run per configuration, and an LLM judge from the same model family. There are no repeated runs of the same configuration, so differences of a question or two shouldn't be read as a ranking. The evaluation set describes private documents, so it can't be reproduced without your own corpus.
- **No rate limiting and no TLS.** The service relies on the API key and an OpenAI spend limit. For a public deployment, put a reverse proxy in front for HTTPS and rate limiting.
- **BM25 index is in memory** and rebuilt at startup, so new documents need an API restart.
- **Streaming traces only the LLM call,** not retrieval, because tracing context doesn't carry across async generator yields.
- **The Docker image is large** (4.72 GB on disk) because of PyTorch and the embedding stack.
- **`langchain-community` (used for `PyPDFLoader`) is being deprecated.** It should be replaced when the PDF parser is upgraded.
- **The `routed` effort policy was evaluated in only one run,** without per-question effort or latency recorded, so its speed and cost benefit is unmeasured.
