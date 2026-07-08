# VeriRAG — Revision & Interview Notes

Personal cheat-sheet for revising the whole project before an interview. Covers: what every
file does, the tricky things I hit (and how they were solved), demo commands for every feature,
and the tech choices with justification.

---

## 1. One-line pitch

> VeriRAG is a **self-correcting, faithfulness-checked RAG service** over PDF documents, wrapped
> in a full **RAGOps** loop: an automated eval benchmark (RAGAS + DeepEval), MLflow experiment
> tracking, and a CI regression gate that blocks any change that makes answers worse.

The headline behaviour: it **retrieves → reranks → generates → judges its own answer → and
abstains ("I don't know") instead of hallucinating** when the evidence isn't there.

---

## 2. Repo map — what each folder / file does

### `app/` — the runtime service
| File | Responsibility |
|---|---|
| `main.py` | FastAPI app. `GET /health`, `POST /query`. Builds the index + graph **once at startup** (lifespan), not per request. |
| `graph.py` | Assembles the LangGraph pipeline (nodes + edges + the two conditional branches: retrieval-retry and faithfulness-retry/abstain). |
| `state.py` | `GraphState` TypedDict — the shared state passed between graph nodes (query, documents, answer, faithfulness_score, abstention_decision, …). |
| `config.py` | All settings via `pydantic-settings`, loaded from `.env` (provider, model names, chunk sizes, eval + MLflow knobs). Single source of truth. |
| `providers.py` | Provider-agnostic **chat-model factory**. `build_chat_model(model, provider=None)` returns a LangChain model for gemini/openai/groq/ollama. Swap LLM with **zero code change**. |
| `llm.py` | Builds the shared model clients (`llm`, `llm_flash`), the embeddings + cross-encoder loaders, `safe_llm_invoke` (retry w/ backoff), and JSON parsers. |
| `ingest.py` | PDF → text (pypdf) → chunks → **Chroma** vector index + **BM25** lexical index. Run once to (re)build the index. |
| `schemas.py` | Pydantic request/response models for the API. |
| `nodes/query_intelligence.py` | Rewrites the query + generates expansions + a "step-back" query (structured output). Improves recall. |
| `nodes/retrieval.py` | **Hybrid retrieval** — runs BM25 (lexical) + vector (semantic) for all query variants, dedupes. |
| `nodes/rerank.py` | Cross-encoder rerank + relevance filter. If top score is low → refine query and retry (bounded). |
| `nodes/compression.py` | Trims retrieved chunks to the relevant bits before generation (less noise, fewer tokens). |
| `nodes/generation.py` | Grounded answer generation from the compressed context. |
| `nodes/faithfulness.py` | LLM **faithfulness judge** (are the answer's claims supported by the context?) + the abstention decision (accept / regenerate / abstain). |

### `eval/` — the RAGOps layer (Weeks 2–3)
| File | Responsibility |
|---|---|
| `benchmark.jsonl` | The **answer key**: 32 hand-curated Q&A rows over the corpus (single_hop / multi_hop / ambiguous / out_of_scope). |
| `schema.py` | Pydantic row schema + `load_benchmark()` validator (unique IDs, valid difficulty, out-of-scope rows carry no context). CLI: `python -m eval.schema`. |
| `runner.py` | Runs the pipeline over the benchmark → scores with **RAGAS** (4 metrics) + **DeepEval** (hallucination) + abstention accuracy → writes `evaluation_report.json` + `failed_cases.csv` → logs to MLflow. |
| `tracking.py` | `log_to_mlflow(report, artifacts)` — params + metrics + artifacts, one run per eval. |
| `gate.py` | **Regression gate**: compares latest report vs `baseline.json`, exits non-zero on regression. `--update` blesses a baseline, `--selftest` checks the logic. |
| `baseline.json` | The committed reference metrics the gate compares against. |
| `requirements-eval.txt` | Eval-only deps (ragas, deepeval, datasets, pandas, mlflow) — kept out of the serving image. |

### Root / infra (Weeks 1 & 4)
| File | Responsibility |
|---|---|
| `.github/workflows/eval-gate.yml` | **CI**: on PR / push, run the benchmark + gate; fail the build on regression. `checks` job (free) + `eval-gate` job (needs `GROQ_API_KEY`). |
| `docker-compose.yml` | `app` + `mlflow` (server) + `postgres` (MLflow backend) for the production-like stack. |
| `Dockerfile` | Container image for the app. |
| `requirements.txt` | Lean **runtime** deps (only what the engine imports). |
| `.env` / `.env.example` | Secrets + config (git-ignored / template). |

---

## 3. The query pipeline, step by step

```
query_intelligence  ── rewrite + expand + step-back query
      ↓
hybrid_retrieve     ── BM25 (lexical) + vector (semantic), deduped
      ↓
rerank              ── cross-encoder scores; filter
   ├── low relevance ─→ Refine_query ─→ back to query_intelligence   (retrieval retry, bounded)
   └── ok ─→ Compress ─→ Generate ─→ Faithfulness Judge ─→ Abstention Threshold
                                                              ├── score low  ─→ regenerate (bounded)
                                                              ├── grounded   ─→ ANSWER + citations
                                                              └── no evidence─→ "I don't know" (abstain)
```

Two self-correction loops = the "Veri" (verified) in VeriRAG: **retrieval retry** (bad context) and
**faithfulness retry / abstain** (unsupported answer).

---

## 4. Gotchas & difficulties I hit (and the fix)

> These are the real "why is this happening?" moments — great interview material because they show
> depth, not just wiring libraries together.

1. **MLflow UI looked empty after logging runs.**
   *Cause:* MLflow 3.x opens the **GenAI** view (Traces = 0); we log **classic runs** (params/metrics).
   *Fix:* toggle to **"Model training"** (top-left) → open `verirag-eval` → Runs table. And always launch the UI against the same store: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

2. **`BadRequestError 400 — 'n' : number must be at most 1` (answer_relevancy broke).**
   *Cause:* RAGAS's `ResponseRelevancy` asks the judge for `n = strictness = 3` generations; **Groq only supports `n=1`**.
   *Fix:* `ResponseRelevancy(strictness=1)` in `run_ragas`.

3. **`faithfulness = None` in the report.**
   *Cause:* Faithfulness is the **heaviest metric** — 2 sequential LLM calls per row (split answer into claims → NLI-verify each against context). Under RAGAS's default 16-way concurrency it hammered Groq's rate limit → every faithfulness job hit `TimeoutError` → all NaN → mean = None.
   *Fix:* throttle with `RunConfig(max_workers=2, timeout=300)`.

4. **Does `max_workers=2` load my laptop more?**
   *Answer:* Only marginally. The real cost is the model inference; `max_workers` just controls how many judge requests are *in flight*. With a **local Ollama** judge, whether they truly run in parallel depends on `OLLAMA_NUM_PARALLEL` (default 1 → the 2nd request queues). `max_workers=1` = gentlest.

5. **Changed `EVAL_JUDGE_MODEL=gemma` but it still ran on Groq.**
   *Cause:* `LLM_PROVIDER` is a **global** provider switch; the model string is just the model *id* within that provider. `gemma4:e4b` is an Ollama tag, so Groq rejected it.
   *Fix:* added **`EVAL_JUDGE_PROVIDER`** so the judge can run on a *different* provider than the pipeline (e.g. judge on local Ollama, generation on Groq).

6. **"Changing `.env` isn't working."**
   *Cause 1:* provider is global — `LLM_MODEL` and `LLM_FLASH_MODEL` share `LLM_PROVIDER`; you can't put them on different providers.
   *Cause 2:* model clients are built **at import time** (`app/llm.py`) — the CLI re-reads `.env` each run (fine), but the **FastAPI server must be restarted** to pick up `.env` edits.

7. **`ImportError: langchain_community.chat_models.vertexai` when importing RAGAS.**
   *Cause:* `langchain-community 0.4` (sunset) removed that module, but every RAGAS version hard-imports it.
   *Fix:* a tiny runtime stub `_patch_legacy_ragas_imports()` in `runner.py` (we never use RAGAS's Vertex path — we pass our own judge).

8. **Terminal spam `[10/12] q010: What is AI-ModelNet?`** — it's just the per-row **progress log** in `collect_predictions` (each row = one full RAG run, can take seconds).

9. **`Evaluating: 3/48` — why 48?** = `answerable_rows × metrics`. `--limit 12` → 12 rows × 4 RAGAS metrics = 48 jobs. (Full 32 → 26 answerable × 4 = 104.) A job ≠ one LLM call (faithfulness = 2).

10. **`6.29s/it`** = tqdm "seconds per iteration" (per job). Flips to `it/s` when items are fast.

11. **Random run names** (`legendary-newt-733`) — MLflow auto-names runs. *Fix:* set `run_name` (now `<timestamp>_<model>`).

12. **CI: "no pipeline ran after I pushed."**
    *Cause:* pushed to branch `v2`, but triggers were `push:[main]` + `pull_request`, and **no `main` branch existed** → no matching event.
    *Fix:* trigger on `[main, v2]` + `workflow_dispatch`; or open a PR into `main`.

13. **Branch mess: `main` looked "reverted".**
    *Cause:* all the work was committed on `v2`; `main` was a **stale** branch. A rename `v2`→`main` **silently fails if `main` already exists**.
    *Fix:* `git checkout v2 && git branch -D main && git branch -m v2 main && git push -u origin main`.

14. **Corpus ↔ benchmark must match.** The benchmark questions must be about the **ingested** PDFs. After swapping PDFs you must re-run `python -m app.ingest`, or the pipeline abstains on everything.

15. **One bad row shouldn't sink a long eval** — `collect_predictions` wraps `graph.invoke` in `try/except`, records the error, and keeps going (rows also surface in `failed_cases.csv` as `pipeline_error`).

---

## 5. Demo commands (feature by feature)

```bash
cd /Users/anideepkalia/Desktop/VeriRAG_v2
# (activate) source .venv/bin/activate   —or prefix each with ./.venv/bin/python
```

**Ingestion — build the search index from PDFs**
```bash
./.venv/bin/python -m app.ingest
```

**Serve the API + query it**
```bash
./.venv/bin/uvicorn app.main:app --reload           # starts on :8000
curl localhost:8000/health
curl -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"What does ANIS stand for?"}'
```

**Eval — validate the benchmark (free, instant)**
```bash
./.venv/bin/python -m eval.schema
```

**Eval — predictions only, no judge cost (shows answering + abstention)**
```bash
./.venv/bin/python -m eval.runner --skip-metrics
```

**Eval — full scored run (RAGAS + DeepEval), auto-logs to MLflow**
```bash
./.venv/bin/python -m eval.runner --limit 12
```

**MLflow — view runs**  *(then switch UI to "Model training" → verirag-eval)*
```bash
./.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

**Regression gate**
```bash
./.venv/bin/python -m eval.gate --update      # set baseline from current run
./.venv/bin/python -m eval.gate               # compare vs baseline → PASS / FAIL(exit 1)
./.venv/bin/python -m eval.gate --selftest    # verify the gate logic
```

**Run the judge locally (free) vs on the cloud**
```bash
# in .env:  EVAL_JUDGE_PROVIDER=ollama  EVAL_JUDGE_MODEL=gemma4:e4b   (pipeline stays on groq)
```

**Full stack via Docker (app + MLflow + Postgres)**
```bash
docker compose up --build
# point the runner at the server:  MLFLOW_TRACKING_URI=http://localhost:5000 python -m eval.runner
```

**CI** — push to `main`/`v2` or open a PR → the `eval-gate` workflow runs (needs `GROQ_API_KEY` secret).

---

## 6. Technologies used & why

| Tech | Used for | Why this one |
|---|---|---|
| **FastAPI + Uvicorn** | HTTP serving | Async, tiny, Pydantic-native request/response validation. |
| **LangGraph** | The RAG pipeline | Needed **stateful, cyclic** control flow (retrieval retry + faithfulness retry + abstain) — a linear chain can't loop back. Conditional edges express the self-correction cleanly. |
| **LangChain** | Retrievers, model wrappers, splitters | Standard integrations so every provider looks the same to the nodes. |
| **Chroma** | Vector store | Embedded, local, zero-infra; persists to disk. |
| **BM25 (rank-bm25)** | Lexical retrieval | Catches exact keywords/IDs that embeddings miss → **hybrid** = better recall. |
| **Cross-encoder (ms-marco-MiniLM)** | Reranking | Joint query-doc scoring is far more accurate than embedding cosine for final ranking. |
| **nomic-embed-text** | Embeddings | Strong open embeddings that run **locally** (no per-call API cost). |
| **Provider factory (Gemini/OpenAI/Groq/Ollama)** | LLM abstraction | Swap models via `.env` only; local (Ollama) for free dev, cloud (Groq) for speed. |
| **Groq (llama-3.3-70b / 3.1-8b-instant)** | Generation + cheap judge | Very fast + cheap cloud inference. |
| **Ollama (gemma)** | Local judge | Free, no rate limits, keeps data local. |
| **RAGAS** | RAG quality metrics | Industry-standard faithfulness / context precision+recall / answer relevancy. |
| **DeepEval** | Hallucination metric | Independent second signal on made-up content. |
| **pydantic / pydantic-settings** | Config + schemas | Typed settings from `.env`, validated request/response + benchmark rows. |
| **MLflow** | Experiment tracking | Every eval run's params/metrics/artifacts, comparable over time in a UI. |
| **Postgres** | MLflow backend (prod) | Durable, multi-user tracking store vs local sqlite for dev. |
| **Docker / docker-compose** | Packaging + local stack | One command brings up app + MLflow + Postgres. |
| **GitHub Actions** | CI regression gate | Runs the benchmark on every PR and **blocks merges that degrade quality**. |
| **pypdf** | PDF text extraction | Lightweight, pure-Python. |

---

## 7. The 4-week arc (the story to tell)

1. **Week 1 — Productionize.** Notebook → a clean FastAPI + LangGraph service (provider-agnostic, config-driven).
2. **Week 2 — Measure.** A hand-curated benchmark + RAGAS/DeepEval runner → objective quality metrics + failing-case export.
3. **Week 3 — Track.** MLflow logging (sqlite/Postgres) + a baseline regression gate → history and a pass/fail signal.
4. **Week 4 — Enforce.** GitHub Actions runs the eval + gate on every PR → quality regressions can't merge.

**Refactor → Measure → Track → Enforce** = the RAGOps loop.

---

## 8. Likely interview questions (quick answers)

- **How do you stop hallucination?** A faithfulness judge scores claim-vs-context support; below threshold after a bounded retry, the system **abstains** instead of answering.
- **Why hybrid retrieval?** BM25 catches exact terms, vectors catch meaning; together recall is higher than either alone; a cross-encoder reranks the union.
- **How do you know a change didn't make it worse?** The benchmark + gate: CI re-runs the eval on every PR and fails on any metric regression beyond tolerance.
- **Why a separate judge model?** Cost — RAGAS/DeepEval make hundreds of calls; a cheap/local judge (`EVAL_JUDGE_MODEL` / `EVAL_JUDGE_PROVIDER`) keeps eval affordable.
- **Biggest gotchas?** Groq's `n=1` cap breaking RAGAS answer_relevancy, and faithfulness timing out under high concurrency — both fixed via metric config + `RunConfig`.
