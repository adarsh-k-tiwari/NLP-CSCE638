def select_consistent_response(cot_paths):
    """
    A simple strategy to select the most common answer among the reasoning paths.
    For a production system, you might compare final answers or use a voting mechanism.
    """
    # For simplicity, assume the final answer is the last non-empty line of the response.
    final_answers = []
    for path in cot_paths:
        lines = [line.strip() for line in path.strip().splitlines() if line.strip()]
        if lines:
            final_answers.append(lines[-1])
    if not final_answers:
        return None, []
    answer_counts = {}
    for ans in final_answers:
        answer_counts[ans] = answer_counts.get(ans, 0) + 1
    selected_answer = max(answer_counts, key=answer_counts.get)
    return selected_answer, final_answers
