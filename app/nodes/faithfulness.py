"""Faithfulness judge + abstention.

The LLM judges each answer claim against the retrieved evidence. The score is the
fraction of supported claims. The answer is accepted only if the score clears the
threshold with no unsupported/contradicted claims; otherwise regenerate (up to a
retry budget) and finally abstain.
"""
import json

from app.config import settings
from app.llm import llm, normalize_list, parse_json_response, safe_llm_invoke

_STATUS_SCORE = {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0, "contradicted": 0.0}


def _clamp(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _normalize_status(status):
    s = str(status).strip().lower().replace(" ", "_").replace("-", "_")
    if s in {"supported", "fully_supported", "entailed"}:
        return "supported"
    if s in {"partial", "partially_supported", "weakly_supported", "mixed"}:
        return "partially_supported"
    if s in {"contradicted", "contradiction", "conflicting"}:
        return "contradicted"
    return "unsupported"


def _evidence_sentences(state):
    sentences = []
    for doc in state.get("documents", []):
        for s in doc.page_content.replace("\n", " ").split("."):
            s = s.strip()
            if len(s) > 10:
                sentences.append(s)
    return sentences


def _score_claims(raw_judgments, claims):
    judged = []
    for item in raw_judgments:
        claim = str(item.get("claim", "")).strip()
        if claim:
            judged.append({"claim": claim, "status": _normalize_status(item.get("status"))})
    # If the judge returned nothing usable, fail safe: treat every claim as unsupported.
    if not judged:
        judged = [{"claim": c, "status": "unsupported"} for c in claims]
    score = round(sum(_STATUS_SCORE[j["status"]] for j in judged) / len(judged), 4)
    return judged, score


def faithfulness_judge_node(state):
    answer = str(state.get("answer", "")).strip()
    claims = normalize_list(state.get("claims", []))

    if not answer or answer == "I don't know":
        return {"faithfulness_score": 0.0, "faithfulness": {}, "unsupported_claims": []}

    if not claims:
        claims = [answer]

    evidence = _evidence_sentences(state)
    evidence_context = "\n".join(f"[E{i}] {s}" for i, s in enumerate(evidence, start=1))

    prompt = f"""You are a strict faithfulness evaluator for a RAG system.

Question:
{state["query"]}

Answer claims:
{json.dumps(claims, indent=2)}

Evidence:
{evidence_context}

Using ONLY the evidence above, judge each claim as one of:
"supported" (evidence directly supports it), "partially_supported" (related but incomplete),
"unsupported" (evidence does not prove it), or "contradicted" (evidence conflicts with it).

Return STRICT JSON:
{{"claim_judgments": [{{"claim": "...", "status": "supported"}}]}}
"""
    judged = parse_json_response(safe_llm_invoke(llm, prompt), {"claim_judgments": []})
    judgments, score = _score_claims(judged.get("claim_judgments") or [], claims)

    unsupported = [j["claim"] for j in judgments if j["status"] == "unsupported"]
    contradicted = [j["claim"] for j in judgments if j["status"] == "contradicted"]

    return {
        "faithfulness_score": score,
        "faithfulness": {
            "claim_judgments": judgments,
            "unsupported_claims": unsupported,
            "contradicted_claims": contradicted,
        },
        "unsupported_claims": unsupported,
    }


def abstention_threshold_node(state):
    answer = str(state.get("answer", "") or "").strip()
    if not answer or answer == "I don't know":
        return {"answer": "I don't know", "claims": [], "abstention_decision": "abstain"}

    score = _clamp(state.get("faithfulness_score", 0.0))
    judge = state.get("faithfulness", {}) or {}
    hard_failure = bool(judge.get("unsupported_claims") or judge.get("contradicted_claims"))
    retry_count = int(state.get("faithfulness_retry_count", 0) or 0)

    if score >= settings.faithfulness_pass_score and not hard_failure:
        return {"abstention_decision": "accept"}

    if retry_count < settings.max_faithfulness_retries:
        return {"abstention_decision": "retry", "faithfulness_retry_count": retry_count + 1}

    return {"answer": "I don't know", "claims": [], "abstention_decision": "abstain"}


def abstention_decision_logic(state):
    return state.get("abstention_decision", "abstain")
