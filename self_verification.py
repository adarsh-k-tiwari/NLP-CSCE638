from generation import generate_response

def verify_answer(query, selected_answer, retrieved_docs):
    """
    Verify the selected answer by asking the model to check the consistency
    of the answer with the evidence.
    """
    evidence_text = "\n".join([f"{doc['title']}: {doc['content']}" for doc in retrieved_docs])
    verification_prompt = (
        f"Verify the following answer based on the provided evidence.\n\n"
        f"Query: {query}\n"
        f"Answer: {selected_answer}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Is the answer consistent and correct based on the evidence? Provide a brief justification."
    )
    verification_response = generate_response(verification_prompt, temperature=0.5)
    return verification_response
