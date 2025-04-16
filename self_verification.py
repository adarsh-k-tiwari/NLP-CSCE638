import logging
from generation import generate_response

# Set up logging for better debugging and tracking
logging.basicConfig(level=logging.INFO)

def verify_answer(query, selected_answer, retrieved_docs, temperature=0.5):
    """
    Verify the selected answer by asking the model to check the consistency
    of the answer with the evidence.
    
    Args:
    query (str): The original question or query.
    selected_answer (str): The answer to verify.
    retrieved_docs (list): List of documents containing evidence.
    temperature (float): The temperature parameter for controlling response randomness.
    
    Returns:
    str: Model's response indicating whether the answer is consistent with the evidence.
    """
    if not retrieved_docs:
        logging.error("No retrieved documents provided.")
        return "Error: No retrieved documents for verification."

    evidence_text = "\n".join([f"{doc['title']}: {doc['content']}" for doc in retrieved_docs])

    # Ensure the prompt is clear and well-structured
    verification_prompt = (
        f"Verify the following answer based on the provided evidence.\n\n"
        f"Query: {query}\n"
        f"Answer: {selected_answer}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Is the answer consistent and correct based on the evidence? Provide a brief justification."
    )

    try:
        # Call the response generation function
        verification_response = generate_response(verification_prompt, temperature=temperature)
        return verification_response
    except Exception as e:
        logging.error(f"Error during response generation: {e}")
        return "Error: Failed to generate verification response."
