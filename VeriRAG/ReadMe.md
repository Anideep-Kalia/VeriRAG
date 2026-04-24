# VeriRAG

VeriRAG is a verification-first Retrieval-Augmented Generation (RAG) notebook built with LangGraph. It does more than retrieve documents and generate an answer: it rewrites the query, retrieves evidence through multiple retrieval strategies, reranks and compresses the evidence, plans the answer before generation, judges whether the final claims are faithful to the retrieved context, and abstains when the system cannot produce a sufficiently grounded answer.

The main notebook is [VeriRAG.ipynb](./VeriRAG.ipynb).

![VeriRAG Architecture](![alt text](image.png))

## Why VeriRAG Exists

Standard RAG systems often fail in subtle ways:

- They retrieve irrelevant or contradictory context.
- They pass too much noisy context into the generator.
- They generate fluent answers that are not actually supported by the retrieved documents.
- They do not know when to say "I don't know."
- They usually treat generation as the final step instead of something that must be verified.

VeriRAG was built to make RAG more trustworthy. It treats every answer as a claim that must survive evidence planning, grounded generation, faithfulness judging, and abstention checks before it is returned.

## What It Does

At a high level, VeriRAG takes a user question and:

1. Rewrites and expands the query.
2. Retrieves candidate documents using both lexical and semantic search.
3. Reranks retrieved documents with a cross-encoder.
4. Retries retrieval when relevance is weak.
5. Compresses retrieved documents down to the most relevant sentences.
6. Builds an explicit answer plan before generation.
7. Generates only from supported claims.
8. Judges every generated claim for faithfulness.
9. Retries generation when faithfulness is too low.
10. Abstains when the answer remains unsafe after retries.

The result is a RAG pipeline that is designed to prefer a grounded, verifiable answer over a confident hallucination.

## Core Workflow

```text
User Question
  -> Query Intelligence
  -> Hybrid Retrieval
  -> Cross-Encoder Reranking
  -> Retrieval Retry Decision
  -> Context Compression
  -> Answer Planning
  -> Plan Quality Decision
  -> Claim-Aware Generation
  -> Faithfulness Judge
  -> Abstention Threshold
       -> accept: return answer
       -> retry: replan and regenerate
       -> abstain: return "I don't know"
```

## Components

### 1. Query Intelligence

The query intelligence node transforms a raw user question into a stronger retrieval query package:

- `rewritten_query`: a cleaner and less ambiguous version of the original question.
- `expanded_queries`: multiple search variations to improve recall.
- `step_back_query`: a broader conceptual query that captures the high-level intent.

This is useful because one user question rarely maps perfectly to the wording used in the source documents.

### 2. Hybrid Retrieval

VeriRAG uses two retrieval methods together:

- BM25 for lexical keyword matching.
- Chroma vector retrieval for embedding-based semantic similarity.

The system runs retrieval over the rewritten query, expanded queries, and step-back query, then deduplicates the retrieved documents. This gives the pipeline both keyword precision and semantic coverage.

### 3. Cross-Encoder Reranking

After retrieval, VeriRAG reranks documents with:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The cross-encoder scores each `(query, document)` pair more precisely than a basic vector similarity score. The notebook then applies relevance thresholds:

- Below the lower threshold: treat retrieval as too weak.
- Between lower and upper threshold: keep the best documents but trigger feedback for retry.
- Above the upper threshold: proceed with the strongest evidence.

This prevents the generator from blindly trusting the first retrieved chunks.

### 4. Retrieval Self-Correction

If retrieval quality is low, the graph loops back through query refinement. The retry prompt includes:

- why retrieval failed,
- previous retrieved snippets,
- previous rewritten query,
- previous expanded queries,
- previous step-back query,
- missing information and unsupported sub-questions.

This lets the system fix retrieval without drifting away from the original user intent.

### 5. Context Compression

Before planning and generation, VeriRAG compresses the retrieved documents. It splits documents into sentences, embeds each sentence, scores sentence similarity against the query, and keeps the most relevant sentences.

This reduces token usage and removes distracting context before the answer is planned.

### 6. Answer Planning

The answer planning node is one of the most important parts of VeriRAG. Instead of directly generating an answer, the system first creates a structured plan:

- sub-questions,
- required information,
- expected factual claims,
- missing information,
- risk flags,
- answer mode,
- evidence mapping,
- supported claims,
- ordered evidence,
- coverage scores.

The planner can set `answer_mode` to `abstain` when the retrieved context is insufficient or conflicting.

### 7. Evidence Mapping

VeriRAG maps each sub-question and expected claim to supporting evidence sentences. It calculates:

- sub-question coverage,
- claim support ratio,
- supported claims,
- unsupported sub-questions,
- risk flags such as `insufficient_evidence` or `unsupported_claims`.

This makes the system inspect whether the answer is possible before trying to generate it.

### 8. Claim-Aware Generation

The generator is restricted to:

- allowed evidence sentences,
- planner-approved supported claims,
- direct and concise answer style.

It returns structured JSON:

```json
{
  "answer": "...",
  "claims": ["...", "..."]
}
```

The notebook filters generated claims against the planner's supported claims. If the generator invents claims outside the approved list, those claims are removed.

### 9. Faithfulness Judge

After generation, VeriRAG sends the answer and its atomic claims to a strict faithfulness judge. The judge evaluates each claim against the available evidence and labels it as:

- `supported`,
- `partially_supported`,
- `unsupported`,
- `contradicted`.

The node computes:

- `faithfulness_score`,
- `evidence_coverage_score`,
- `retrieval_confidence`,
- `final_confidence`,
- `confidence_label`,
- unsupported claims,
- weakly supported claims,
- contradicted claims.

This turns answer validation into a first-class step in the graph, not a manual afterthought.

### 10. Faithfulness Retry Loop

If the Faithfulness Judge finds that an answer is not faithful enough, the graph does not immediately fail. It sends the judge output back into the planner.

The planner receives:

- previous answer,
- previous claims,
- full judge analysis,
- faithfulness score,
- unsupported claims,
- weakly supported claims,
- contradicted claims.

Then it replans and regenerates. The current retry limit is:

```python
MAX_FAITHFULNESS_RETRIES = 2
```

This creates a controlled self-correction loop for generation quality.

### 11. Abstention Threshold

The abstention threshold node receives the result of the Faithfulness Judge and makes the final decision:

- `accept`: the answer is faithful enough and has no hard failures.
- `retry`: faithfulness is too low but retries remain.
- `abstain`: the answer is empty, already abstained, or still unsafe after max retries.

The threshold is currently:

```python
ABSTENTION_FAITHFULNESS_THRESHOLD = 0.85
```

If the answer fails after all retries, VeriRAG returns:

```text
I don't know
```

The rejected answer is still stored in the abstention metadata for debugging.

## What Makes VeriRAG Different

### Verification Is Built Into The Graph

Many RAG systems stop at generation. VeriRAG makes verification part of the workflow through planning, faithfulness judging, retry routing, and abstention.

### It Plans Before It Answers

The answer planner forces the system to ask: "What claims can I safely make from this evidence?" This reduces hallucination before generation even starts.

### It Generates From Claims, Not Just Context

The generator is not free to use every retrieved sentence. It is constrained to planner-approved supported claims and ordered evidence.

### It Has Two Self-Correction Loops

VeriRAG can correct both retrieval and generation:

- Retrieval retry loop: improves the query when retrieved evidence is weak.
- Faithfulness retry loop: replans and regenerates when the answer is not grounded enough.

### It Knows When To Abstain

The abstention threshold node is intentionally conservative. If the answer cannot pass verification, the system says "I don't know" instead of returning a polished but unsupported answer.

### It Combines Multiple Evidence Signals

The confidence score is not based on one metric. It combines:

- claim-level faithfulness,
- planner evidence coverage,
- retrieval confidence.

This gives a more complete picture of answer quality.

### It Includes A Realistic Noisy Enterprise Corpus

The included `data.txt` is a synthetic enterprise knowledge base with HR, IT, security, finance, product, customer support, operations, and compliance records. It also includes outdated memos, unapproved drafts, incorrect notes, and irrelevant facts. This makes the notebook a better testbed for real RAG failure modes instead of only clean-document demos.


## Graph Nodes

| Node | Purpose |
| --- | --- |
| `query_intelligence` | Rewrites, expands, and generalizes the user query. |
| `hybrid_retrieve` | Retrieves documents using BM25 and vector search. |
| `rerank` | Reranks documents with a cross-encoder and applies relevance thresholds. |
| `Refine_query` | Increments retrieval retry count and sends the flow back to query intelligence. |
| `Compress` | Keeps only the most relevant sentences from the top documents. |
| `Plan` | Builds a structured answer plan and evidence map. |
| `Claim aware generation` | Generates an answer using only supported claims. |
| `Faithfulness Judge` | Scores each answer claim against the evidence. |
| `Abstention Threshold` | Accepts, retries, or abstains based on judge output. |

## Key Configuration

```python
FAITHFULNESS_PASS_SCORE = 0.85
ABSTENTION_FAITHFULNESS_THRESHOLD = FAITHFULNESS_PASS_SCORE
MAX_FAITHFULNESS_RETRIES = 2
```

Reranking thresholds inside the notebook:

```python
upper_threshold = 0.7
lower_threshold = 0.4
```

Retrieval retry limit:

```python
max_iterations = 2
```

## Tech Stack

- LangGraph for graph orchestration.
- LangChain for prompts, documents, retrievers, and model interfaces.
- Google Vertex AI / Gemini through `langchain-google-vertexai`.
- Chroma for vector storage.
- Hugging Face embeddings with `nomic-ai/nomic-embed-text-v1`.
- BM25 for lexical retrieval.
- CrossEncoder reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- scikit-learn cosine similarity for sentence ranking and compression.

## Repository Files

| File | Description |
| --- | --- |
| `VeriRAG.ipynb` | Main notebook containing the full VeriRAG workflow. |
| `data.txt` | Synthetic enterprise knowledge base with realistic policies, support docs, noisy records, and contradictions. |
| `requirements.txt` | Python dependencies used by the notebook. |
| `VeriRAG Architecture.jpeg` | Architecture diagram for the workflow. |
| `backup/` | Older notebook experiments and previous versions. |

## How To Run

1. Install dependencies:

```bash
pip install -r requirements.txt
pip install langchain-google-vertexai python-dotenv
```

2. Configure Google Cloud authentication for Vertex AI:

```bash
gcloud auth application-default login
```

3. Add any required environment variables in `.env`, such as LangSmith or Hugging Face tokens if you use tracing or private models.

4. Open and run:

```text
VeriRAG.ipynb
```

5. Modify the query in the final inference cell:

```python
result = graph.invoke({
    "query": "How many vacation days do full-time employees receive?",
    "iteration": 0,
    "faithfulness_retry_count": 0
})
```

## Final Output

The final result includes:

- answer,
- claims,
- planner mode,
- planner coverage,
- faithfulness score,
- confidence label,
- faithfulness retry count,
- abstention decision,
- abstention details,
- faithfulness details.

Example fields:

```python
result.get("answer")
result.get("claims")
result.get("faithfulness_score")
result.get("abstention_decision")
result.get("abstention")
```

## Current Limitations

- The notebook uses a synthetic enterprise corpus for demonstration.
- It currently loads documents from `data.txt`, not from a production document pipeline or document store.
- The Vertex AI project and model configuration are hardcoded in the notebook and should be parameterized for production.
- Thresholds are heuristic and should be tuned against a real evaluation set.
- The faithfulness judge is still an LLM-based evaluator, so high-stakes deployments should add deterministic checks and human review where appropriate.

## Future Improvements

- Add document metadata and citation display.
- Replace the demo corpus with a scalable ingestion pipeline.
- Add evaluation datasets for threshold tuning.
- Store intermediate graph traces for easier debugging.
- Add unit tests for routing logic.
- Expose the workflow through an API or lightweight UI.
- Add deterministic citation validation for each generated claim.

## Summary

VeriRAG is a RAG system designed around one principle: an answer should not be returned just because it sounds good. It should be retrieved, planned, generated, judged, retried when possible, and rejected when necessary.

That makes VeriRAG a stronger foundation for grounded question answering than a basic retrieve-and-generate pipeline.
