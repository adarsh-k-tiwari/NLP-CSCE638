import openai
from config import OPENAI_API_KEY, LLM_MODEL

openai.api_key = OPENAI_API_KEY

def generate_response(prompt, temperature=0.7):
    """
    Call the LLM API with the given prompt and return the response text.
    """
    response = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=300
    )
    return response["choices"][0]["message"]["content"]

def generate_cot_paths(prompt, num_paths=5):
    """
    Generate multiple chain-of-thought reasoning paths.
    """
    responses = []
    for _ in range(num_paths):
        responses.append(generate_response(prompt, temperature=0.8))
    return responses
