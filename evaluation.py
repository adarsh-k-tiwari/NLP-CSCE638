def evaluate_response(generated_answer, ground_truth):
    """
    A simple evaluation function comparing generated answer to ground truth.
    For a real application, you might implement more sophisticated metrics.
    """
    return generated_answer.strip().lower() == ground_truth.strip().lower()

def run_evaluation(test_cases):
    """
    Run evaluation over a list of test cases.
    Each test case is a tuple: (query, ground_truth)
    """
    results = []
    for query, ground_truth in test_cases:
        # In a full implementation, you would run the entire pipeline.
        # Here we use a placeholder.
        generated_answer = "placeholder answer"  # Replace with pipeline call
        is_correct = evaluate_response(generated_answer, ground_truth)
        results.append({"query": query, "correct": is_correct})
    return results
