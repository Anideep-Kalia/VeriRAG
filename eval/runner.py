"""Evaluation runner: run the pipeline over the benchmark, score it, persist results.

  python -m eval.runner [--limit N] [--skip-metrics]

Stages:
  1. Predictions   — run build_graph() per row, capture answer/contexts/abstained.
  2. RAGAS         — faithfulness, context_precision, context_recall, answer_relevancy
                     (answerable rows only; cheap judge via settings.eval_judge_model).
  3. DeepEval      — hallucination rate on answered rows.
  4. Abstention    — fraction of out_of_scope rows correctly abstained.
  5. Persist       — evaluation_report.json + failed_cases.csv.

COST: stages 1-3 hit paid LLMs. Always dev with --limit first (a full 100-row run is
hundreds of judge calls). --skip-metrics runs the pipeline only (no judge calls).
"""
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

from app.config import settings
from eval.schema import load_benchmark

REPORT_PATH = Path("evaluation_report.json")
FAILED_PATH = Path("failed_cases.csv")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Predictions
# ──────────────────────────────────────────────────────────────────────────────

# parser to noramlize the LLM outputs to a single string
def _answer_text(answer):
    if isinstance(answer, list):
        return " ".join(item.get("text", "") for item in answer).strip()
    return str(answer or "").strip()


# RAG run against rows of questions present in benchmark.jsonl
def collect_predictions(rows):
    from app.graph import build_graph

    graph = build_graph()
    preds = []
    for i, row in enumerate(rows, start=1):
        print(f"[{i}/{len(rows)}] {row.id}: {row.question[:70]}")
        try:
            result = graph.invoke({
                "query": row.question,
                "iteration": 0,
                "faithfulness_retry_count": 0,
            })
        except Exception as e:
            # One bad row shouldn't sink a long run — record it and keep going.
            print(f"    !! pipeline error, skipping row: {e}")
            result = {"error": str(e)}
        contexts = [d.page_content for d in (result.get("documents") or [])]
        preds.append({
            "id": row.id,
            "question": row.question,
            "ground_truth": row.ground_truth,
            "ground_truth_contexts": row.ground_truth_contexts,
            "difficulty": row.difficulty,
            "answer": _answer_text(result.get("answer", "")),
            "contexts": contexts,
            "abstained": result.get("abstention_decision") == "abstain",
            "answer_source": result.get("answer_source", ""),
            "error": result.get("error"),
        })
    return preds


# ──────────────────────────────────────────────────────────────────────────────
# 2. RAGAS
# ──────────────────────────────────────────────────────────────────────────────

def _judge_model():
    from app.providers import build_chat_model

    return build_chat_model(settings.eval_judge_model, provider=settings.eval_judge_provider or None)


def _patch_legacy_ragas_imports():
    """ragas hard-imports langchain_community.chat_models.vertexai, which was
    removed in langchain-community 0.4+. We never use ragas' Vertex path (we pass
    our own judge), so stub it just enough to let ragas import."""
    import sys
    import types

    mod = "langchain_community.chat_models.vertexai"
    if mod not in sys.modules:
        stub = types.ModuleType(mod)
        stub.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[mod] = stub
    import langchain_community.llms as L
    if not hasattr(L, "VertexAI"):
        L.VertexAI = type("VertexAI", (), {})


def run_ragas(answerable):
    """Returns (per_row: dict[id]->dict, aggregate: dict). Empty on failure."""
    if not answerable:
        return {}, {}
    try:
        _patch_legacy_ragas_imports()
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.run_config import RunConfig
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )

        from app.llm import get_embeddings

        samples = [{
            "user_input": p["question"],
            "response": p["answer"] or " ",
            "retrieved_contexts": p["contexts"] or [" "],
            "reference": p["ground_truth"],
        } for p in answerable]

        ds = EvaluationDataset.from_list(samples)
        metrics = [
            Faithfulness(),
            ResponseRelevancy(strictness=2),  # Groq allows n<=1; default strictness=3 sends n=3 -> 400
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ]
        result = evaluate(
            ds,
            metrics=metrics,
            llm=LangchainLLMWrapper(_judge_model()),
            embeddings=LangchainEmbeddingsWrapper(get_embeddings()),
            # ponytail: 2 workers dodges Groq rate-limit timeouts; raise if you have a paid tier
            run_config=RunConfig(max_workers=2, timeout=300),
        )
        df = result.to_pandas()
    except Exception as e:
        print(f"[ragas] skipped: {e}")
        return {}, {}

    # RAGAS renames its output columns between versions 
    wanted = {
        "faithfulness": "faithful",
        "answer_relevancy": "relevan",
        "context_precision": "precision",
        "context_recall": "recall",
    }
    colmap = {}
    for key, needle in wanted.items():
        for col in df.columns:
            if needle in col.lower():
                colmap[key] = col
                break

    per_row = {}
    for idx, p in enumerate(answerable):
        per_row[p["id"]] = {
            key: _safe_float(df.iloc[idx][col]) for key, col in colmap.items()
        }
    aggregate = {key: _safe_float(df[col].mean()) for key, col in colmap.items()}
    return per_row, aggregate


def _safe_float(v):
    try:
        f = float(v)
        return None if f != f else round(f, 4)   # f != f catches NaN
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 3. DeepEval hallucination
# ──────────────────────────────────────────────────────────────────────────────

def run_deepeval(answerable):
    """Mean hallucination score over answered rows with reference context. None on failure."""
    scored = [
        p for p in answerable
        if not p["abstained"] and p["ground_truth_contexts"]
    ]
    if not scored:
        return None, {}
    try:
        _patch_legacy_ragas_imports()
        from deepeval.metrics import HallucinationMetric
        from deepeval.models.base_model import DeepEvalBaseLLM
        from deepeval.test_case import LLMTestCase

        judge = _judge_model()

        class _Judge(DeepEvalBaseLLM):
            def load_model(self):
                return judge

            def generate(self, prompt, schema=None):
                text = judge.invoke(prompt).content
                if isinstance(text, list):
                    text = " ".join(i.get("text", "") for i in text)
                if schema is not None:
                    return schema.model_validate_json(_extract_json(text))
                return text

            async def a_generate(self, prompt, schema=None):
                return self.generate(prompt, schema)

            def get_model_name(self):
                return "verirag-eval-judge"

        metric = HallucinationMetric(threshold=0.5, model=_Judge())
        per_row = {}
        for p in scored:
            tc = LLMTestCase(
                input=p["question"],
                actual_output=p["answer"],
                context=p["ground_truth_contexts"],
            )
            metric.measure(tc)
            per_row[p["id"]] = _safe_float(metric.score)
    except Exception as e:
        print(f"[deepeval] skipped: {e}")
        return None, {}

    vals = [v for v in per_row.values() if v is not None]
    rate = round(sum(vals) / len(vals), 4) if vals else None
    return rate, per_row


def _extract_json(text):
    text = str(text).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


# ──────────────────────────────────────────────────────────────────────────────
# 4/5. Aggregate, abstention, persist
# ──────────────────────────────────────────────────────────────────────────────

def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def write_failed_cases(preds, ragas_rows, deepeval_rows, threshold):
    rows = []
    for p in preds:
        rid = p["id"]
        rg = ragas_rows.get(rid, {})
        hall = deepeval_rows.get(rid)
        reasons = []
        if p.get("error"):
            reasons.append("pipeline_error")
        if p["difficulty"] == "out_of_scope" and not p["abstained"]:
            reasons.append("answered_out_of_scope")
        for metric in ("faithfulness", "context_recall", "answer_relevancy", "context_precision"):
            v = rg.get(metric)
            if v is not None and v < threshold:
                reasons.append(f"{metric}<{threshold}")
        if hall is not None and hall > (1 - threshold):
            reasons.append("high_hallucination")
        if reasons:
            rows.append({
                "id": rid,
                "difficulty": p["difficulty"],
                "abstained": p["abstained"],
                "faithfulness": rg.get("faithfulness"),
                "context_recall": rg.get("context_recall"),
                "answer_relevancy": rg.get("answer_relevancy"),
                "context_precision": rg.get("context_precision"),
                "hallucination": hall,
                "reasons": "; ".join(reasons),
                "question": p["question"],
                "answer": p["answer"][:300],
            })

    with FAILED_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "difficulty", "abstained", "faithfulness", "context_recall",
            "answer_relevancy", "context_precision", "hallucination", "reasons",
            "question", "answer",
        ])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def run(limit=None, skip_metrics=False, track=True):
    rows = load_benchmark()
    if limit:
        rows = rows[:limit]

    preds = collect_predictions(rows)
    answerable = [p for p in preds if p["difficulty"] != "out_of_scope"]
    oos = [p for p in preds if p["difficulty"] == "out_of_scope"]
    abstention_accuracy = (
        round(sum(p["abstained"] for p in oos) / len(oos), 4) if oos else None
    )

    if skip_metrics:
        ragas_rows, ragas_agg, hallucination_rate, deepeval_rows = {}, {}, None, {}
    else:
        ragas_rows, ragas_agg = run_ragas(answerable)
        hallucination_rate, deepeval_rows = run_deepeval(answerable)

    metrics = {
        "faithfulness": ragas_agg.get("faithfulness"),
        "context_precision": ragas_agg.get("context_precision"),
        "context_recall": ragas_agg.get("context_recall"),
        "answer_relevancy": ragas_agg.get("answer_relevancy"),
        "hallucination_rate": hallucination_rate,
        "abstention_accuracy": abstention_accuracy,
    }

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git_commit(),
        "n_rows": len(preds),
        "n_answerable": len(answerable),
        "n_out_of_scope": len(oos),
        "params": {
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "retrieval_strategy": "hybrid_bm25_vector",
            "chunk_size": settings.chunk_size,
            "eval_judge_model": settings.eval_judge_model,
        },
        "metrics": metrics,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    n_failed = write_failed_cases(preds, ragas_rows, deepeval_rows, settings.eval_fail_threshold)

    if track:
        try:
            from eval.tracking import log_to_mlflow
            run_id = log_to_mlflow(report, [REPORT_PATH, FAILED_PATH])
            print(f"Logged run {run_id[:8]} to MLflow experiment '{settings.mlflow_experiment}'.")
        except Exception as e:
            print(f"[mlflow] skipped: {e}")

    print("\n=== Aggregate metrics ===")
    for k, v in metrics.items():
        print(f"  {k:20} {v}")
    print(f"\nWrote {REPORT_PATH} and {FAILED_PATH} ({n_failed} failed cases).")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="evaluate only first N rows")
    parser.add_argument("--skip-metrics", action="store_true", help="predictions only, no judge calls")
    parser.add_argument("--no-track", action="store_true", help="skip MLflow logging")
    args = parser.parse_args(argv)
    run(limit=args.limit, skip_metrics=args.skip_metrics, track=not args.no_track)


if __name__ == "__main__":
    main()
