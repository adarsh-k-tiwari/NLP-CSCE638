from generation import generate_response
import logging

# Set up logging for better debugging and tracking
logging.basicConfig(level=logging.INFO)

def generate_verification_prompt(query, selected_answer, retrieved_docs):
    """
    Helper function to generate the verification prompt with retrieved evidence.

    Args:
    query (str): The original question or query.
    selected_answer (str): The answer to verify.
    retrieved_docs (list): List of documents containing evidence.

    Returns:
    str: The formatted prompt.
    """
    # Join all the retrieved documents into evidence text
    evidence_text = "\n".join([f"{doc['title']}: {doc['content']}" for doc in retrieved_docs])

    # Create the final verification prompt
    verification_prompt = (
        f"Verify the following answer based on the provided evidence.\n\n"
        f"Query: {query}\n"
        f"Answer: {selected_answer}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Is the answer consistent and correct based on the evidence? Provide a brief justification."
    )

    return verification_prompt


def verify_answer(query, selected_answer, retrieved_docs, temperature=0.5, max_retries=3):
    """
    Verify the selected answer by asking the model to check the consistency
    of the answer with the evidence, allowing for multiple retries and better error handling.
    
    Args:
    query (str): The original question or query.
    selected_answer (str): The answer to verify.
    retrieved_docs (list): List of documents containing evidence.
    temperature (float): The temperature parameter for controlling response randomness.
    max_retries (int): Number of retries in case of failure.

    Returns:
    str: Model's response indicating whether the answer is consistent with the evidence.
    """
    if not retrieved_docs:
        logging.error("No retrieved documents provided.")
        return "Error: No retrieved documents for verification."

    verification_prompt = generate_verification_prompt(query, selected_answer, retrieved_docs)

    for attempt in range(max_retries):
        try:
            # Call the response generation function
            verification_response = generate_response(verification_prompt, temperature=temperature)
            return verification_response
        except Exception as e:
            logging.error(f"Error during response generation on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return "Error: Failed to generate verification response after multiple attempts."
            else:
                logging.info(f"Retrying due to error... ({attempt + 1}/{max_retries})")
                continue


# Example of calling the function
query = "What is the capital of France?"
selected_answer = "Paris"
retrieved_docs = [
    {"title": "Document 1", "content": "The capital of France is Paris, located in the northern part."},
    {"title": "Document 2", "content": "Paris is known for its culture and history, including the Eiffel Tower."},
]

response = verify_answer(query, selected_answer, retrieved_docs)
print("Verification Response:", response)