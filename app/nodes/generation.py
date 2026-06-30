"""Grounded answer generation — answer strictly from retrieved docs, or abstain."""
import json

from app.llm import llm, normalize_list, parse_json_response, safe_llm_invoke


def _references(docs):
    """One citation per (source, page) of the docs handed to the generator."""
    refs, seen = [], set()
    for d in docs:
        meta = d.metadata or {}
        key = (meta.get("source", "unknown"), meta.get("page", 0))
        if key not in seen:
            seen.add(key)
            refs.append({"document": meta.get("source", "unknown"), "page": meta.get("page", 0) + 1})
    return refs


def generate_answer_node(state):
    docs = state.get("documents", [])
    if not docs:
        return {"answer": "I don't know", "claims": [], "references": []}

    context = "\n\n".join(f"[DOC_{i}] {d.page_content}" for i, d in enumerate(docs, start=1))

    # On a faithfulness retry, steer away from the claims the judge rejected.
    retry_guidance = ""
    unsupported = (state.get("faithfulness", {}) or {}).get("unsupported_claims") or []
    if state.get("faithfulness_retry_count", 0) and unsupported:
        retry_guidance = (
            "\nThe previous answer failed the faithfulness check. "
            "Do NOT repeat these unsupported claims:\n" + json.dumps(unsupported, indent=2)
        )

    prompt = f"""You are a strict grounded answer generator.

Question:
{state["query"]}

Context:
{context}
{retry_guidance}

Instructions:
- Use ONLY the context above.
- Answer directly and concisely.
- Keep claims atomic and factual.
- If the context does not contain the answer, return "I don't know".

Return STRICT JSON:
{{"answer": "...", "claims": ["...", "..."]}}
"""
    result = parse_json_response(safe_llm_invoke(llm, prompt), {"answer": "I don't know", "claims": []})
    answer = str(result.get("answer", "")).strip() or "I don't know"
    if answer == "I don't know":
        return {"answer": "I don't know", "claims": [], "references": []}

    claims = normalize_list(result.get("claims")) or [answer]
    return {"answer": answer, "claims": claims, "references": _references(docs)}
