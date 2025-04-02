def build_cot_prompt(query, retrieved_docs):
    """
    Build a chain-of-thought prompt that includes the query and the retrieved evidence.
    """
    evidence_text = "\n".join(
        [f"Title: {doc['title']}\nContent: {doc['content']}" for doc in retrieved_docs]
    )
    prompt = (
        f"Answer the following query with step-by-step reasoning (Chain-of-Thought) "
        f"based on the evidence provided:\n\n"
        f"Query: {query}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Chain-of-Thought:"
    )
    return prompt
