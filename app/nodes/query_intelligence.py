"""Query rewriting & expansion node (verbatim logic; imports rewired)."""
from langchain_core.prompts import ChatPromptTemplate

from app.llm import safe_llm_invoke, structured_llm_flash

optimizer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert query optimizer. Return ONLY JSON."),
    ("human", """
        User Query:
        {query}

        Perform:
        1. Rewrite query
        2. Generate 3 variations
        3. Generate step-back query

        Return JSON:

        {{
        "rewritten_query": "...",
        "expanded_queries": ["...", "...", "...", "..."],
        "step_back_query": "..."
        }}
    """)
])


def query_intelligence_node(state):

    iteration = state.get("iteration", 0)

    if iteration == 0:
        query = state["query"]
        prompt = optimizer_prompt.invoke({"query": query})
    else:
        feedback = state.get("retrieval_feedback", {})
        snippets = [
            doc.page_content[:200]
            for doc in state.get("documents", [])[:2]
        ]

        refined_query = f"""
            Previous retrieval failed.

            Reason: {feedback.get("reason")}
            Max Score: {feedback.get("max_score")}
            Avg Score: {feedback.get("avg_score")}
            Missing Information: {feedback.get("missing_information")}
            Unsupported Sub-questions: {feedback.get("unsupported_sub_questions")}
            Coverage: {feedback.get("coverage")}

            The following snippets were retrieved but are NOT relevant:
            {chr(10).join(snippets)}

            IMPORTANT:
            - These snippets are incorrect or irrelevant
            - Do NOT base your query on them
            - Use them only to understand what went wrong

            Original Query:
            {state["query"]}

            Previous Rewritten Query:
            {state.get("rewritten_query")}

            Previous Expanded Query:
            {state.get("expanded_queries")}

            Previous Step_back Query:
            {state.get("step_back_query")}

            Your task:
            - Identify why retrieval failed
            - Fix the query without drifting away from user intent
            - Preserve original intent strictly
            - Improve specificity and keywords
        """
        prompt = optimizer_prompt.invoke({"query": refined_query})

    result = safe_llm_invoke(structured_llm_flash, prompt)

    return {
        "rewritten_query": result.rewritten_query,
        "expanded_queries": result.expanded_queries,
        "step_back_query": result.step_back_query
    }
