import os
import json
import logging
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
import wikipediaapi
from dotenv import load_dotenv

# API in .env
load_dotenv()

def generate_response(prompt, temperature=0.5, model_name="gpt-3.5"):
    if model_name == "gpt-3.5":
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set in .env")
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"GPT-3.5 error: {str(e)}")
            return f"Mock GPT-3.5 response: {prompt[:50]}, (Error: {str(e)})"
    elif model_name == "llama-2":
        try:
            from transformers import pipeline
            pipe = pipeline(
                "text-generation",
                model="meta-ai/llama-2-7b",
                device=0,
                token=os.getenv("HF_TOKEN") or None
            )
            response = pipe(prompt, max_length=512, temperature=temperature)
            return response[0]["generated_text"]
        except Exception as e:
            logger.error(f"LLaMA-2 error: {str(e)}")
            return f"Mock LLaMA-2 response: {prompt[:50]}, (Error: {str(e)})"
    elif model_name == "deepseek":
        try:
            from transformers import pipeline
            pipe = pipeline("text-generation", model="DeepSeek/DeepSeek-R1", device=0) 
            response = pipe(prompt, max_length=512, temperature=temperature)
            return response[0]["generated_text"]
        except Exception as e:
            logger.error(f"DeepSeek error: {str(e)}")
            return f"Mock DeepSeek response: {prompt[:50]}, (Error: {str(e)})"
    else:
        raise ValueError(f"Unknown model: {model_name}")

def verify_answer(query, selected_answer, retrieved_docs, model_name="gpt-3.5"):

    evidence_text = "\n".join([f"{doc['title']}: {doc['content']}" for doc in retrieved_docs])
    verification_prompt = (
        f"Verify the following answer based on the provided evidence.\n\n"
        f"Query: {query}\n"
        f"Answer: {selected_answer}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Is the answer consistent and correct based on the evidence? Provide a brief justification."
    )
    verification_response = generate_response(verification_prompt, temperature=0.5, model_name=model_name)
    return "consistent and correct" in verification_response.lower(), verification_response

def retrieve_documents(query, dataset_name="FEVER"):
    """
    Retrieve documents using Wikipedia for FEVER, fallback for others.
    """
    wiki = wikipediaapi.Wikipedia("en")
    if dataset_name == "FEVER":
        page_title = query.split()[0]
        page = wiki.page(page_title)
        if page.exists():
            return [{"title": page.title, "content": page.summary[:500]}]
    page = wiki.page(query.split()[0])
    if page.exists():
        return [{"title": page.title, "content": page.summary[:500]}]
    return [{"title": "No Doc", "content": f"No relevant information found for {query[:50]}..."}]

def preprocess_dataset(dataset_name, max_samples=100):
    """
    Load and preprocess dataset, returning queries, answers, and labels.
    """
    if dataset_name == "HaluEval":
        halueval_path = "/Users/hwiyoonkim/Desktop/HwiyoonKim/NLP-CSCE638/halueval.json"
        if not os.path.exists(halueval_path):
            logger.warning(f"HaluEval dataset not found at {halueval_path}. Using mock data.")
            data = [
                {"query": "Is the moon made of cheese?", "answer": "No", "label": "Correct"},
                {"query": "Does 2+2=22?", "answer": "No", "label": "Correct"}
            ]
        else:
            with open(halueval_path, "r") as f:
                data = json.load(f)[:max_samples]
        queries = [item["query"] for item in data]
        true_answers = [item["answer"] for item in data]
        labels = [item["label"] for item in data]
    elif dataset_name == "TruthfulQA":
        data = load_dataset("truthful_qa", "generation", split="validation")[:max_samples]
        queries = [item["question"] for item in data]
        true_answers = [item["best_answer"] for item in data]
        labels = [1 if item["correct_answers"] else 0 for item in data]
    elif dataset_name == "FEVER":
        data = load_dataset("fever", "v1.0", split="test")[:max_samples]
        queries = [item["claim"] for item in data]
        true_answers = [item["evidence"] for item in data]
        labels = [item["label"] for item in data]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    queries = [tokenizer.decode(tokenizer.encode(q, max_length=512, truncation=True)) for q in queries]
    true_answers = [tokenizer.decode(tokenizer.encode(a, max_length=512, truncation=True)) for a in true_answers]

    return queries, true_answers, labels

def evaluate_self_verification(model_name, dataset_name, max_samples=100):
    try:
        queries, true_answers, labels = preprocess_dataset(dataset_name, max_samples)
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
        return 0.0

    predicted_labels = []
    correct_count = 0

    for i, (query, true_answer, label) in enumerate(zip(queries, true_answers, labels)):
        logger.info(f"Processing {dataset_name} sample {i+1}/{len(queries)} for {model_name}...")
        selected_answer = generate_response(query, temperature=0.5, model_name=model_name)
        retrieved_docs = retrieve_documents(query, dataset_name)
        is_correct, _ = verify_answer(query, selected_answer, retrieved_docs, model_name)

        if dataset_name == "HaluEval":
            predicted_label = "Correct" if is_correct else "Incorrect"
            correct = predicted_label == label
        elif dataset_name == "TruthfulQA":
            predicted_label = 1 if is_correct else 0
            correct = predicted_label == label
        elif dataset_name == "FEVER":
            predicted_label = selected_answer if is_correct else "NOT ENOUGH INFO"
            correct = predicted_label == label
        else:
            correct = False

        predicted_labels.append(predicted_label)
        if correct:
            correct_count += 1

    accuracy = correct_count / len(queries)
    return round(accuracy * 100, 2) 

def main():
    models = ["gpt-3.5", "llama-2", "deepseek"]
    datasets = ["HaluEval", "TruthfulQA", "FEVER"]
    max_samples = 100
    results = []

    for model in models:
        for dataset in datasets:
            logger.info(f"Evaluating {model} on {dataset}")
            accuracy = evaluate_self_verification(model, dataset, max_samples)
            results.append({
                "Model": model,
                "Dataset": dataset,
                "Self-Verification Accuracy (%)": accuracy
            })
            logger.info(f"{model} on {dataset}: Accuracy = {accuracy:.2f}%")

    print("\nSelf-Verification Accuracy Chart (%):")
    results_df = pd.DataFrame(results)
    pivot_table = results_df.pivot(index="Model", columns="Dataset", values="Self-Verification Accuracy (%)")
    print(pivot_table.to_string(float_format="%.2f"))

if __name__ == "__main__":
    main()