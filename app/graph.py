"""LangGraph assembly.

retrieve -> rerank (retry on low relevance) -> compress -> generate -> judge
-> accept | regenerate (faithfulness retry) | abstain.
"""
from langgraph.graph import END, StateGraph

from app.nodes.compression import compress_documents_node
from app.nodes.faithfulness import (
    abstention_decision_logic,
    abstention_threshold_node,
    faithfulness_judge_node,
)
from app.nodes.generation import generate_answer_node
from app.nodes.query_intelligence import query_intelligence_node
from app.nodes.rerank import Refine_query_node, Retry_decision_logic, rerank_and_filter_node
from app.nodes.retrieval import hybrid_retrieve_node
from app.state import GraphState


def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("query_intelligence", query_intelligence_node)
    builder.add_node("hybrid_retrieve", hybrid_retrieve_node)
    builder.add_node("rerank", rerank_and_filter_node)
    builder.add_node("Refine_query", Refine_query_node)
    builder.add_node("Compress", compress_documents_node)
    builder.add_node("Generate", generate_answer_node)
    builder.add_node("Faithfulness Judge", faithfulness_judge_node)
    builder.add_node("Abstention Threshold", abstention_threshold_node)

    builder.set_entry_point("query_intelligence")
    builder.add_edge("query_intelligence", "hybrid_retrieve")
    builder.add_edge("hybrid_retrieve", "rerank")
    builder.add_conditional_edges(
        "rerank", Retry_decision_logic, {"retry": "Refine_query", "generate": "Compress"}
    )
    builder.add_edge("Refine_query", "query_intelligence")
    builder.add_edge("Compress", "Generate")
    builder.add_edge("Generate", "Faithfulness Judge")
    builder.add_edge("Faithfulness Judge", "Abstention Threshold")
    builder.add_conditional_edges(
        "Abstention Threshold",
        abstention_decision_logic,
        {"retry": "Generate", "accept": END, "abstain": END},
    )

    return builder.compile()
