import pandas as pd
import json
import os
from datasets import load_dataset




def clean_text(text):
    if isinstance(text, str):
        text = text.lower().strip()
        cleaned_text = text.replace('\n', ' ').replace('\r', '').strip()
    return cleaned_text

def load_truthfulQA():
    dataset = load_dataset("truthfulqa/truthful_qa", "generation")
    df = pd.DataFrame(dataset['train'])
    df['question'] = df['question'].apply(clean_text)
    df['source'] = df['source'].apply(clean_text)
    df['best_answer'] = df['best_answer'].apply(clean_text)
    df['correct_answers'] = df['correct_answers'].apply(clean_text)
    df['incorrect_answer'] = df['incorrect_answer'].apply(clean_text)
    return df

def load_HaluEval():
    dataset = load_dataset("pminervini/HaluEval", "qa_samples")
    df = pd.DataFrame(dataset['data'])
    df['knowledge'] = df['knowledge'].apply(clean_text)
    df['question'] = df['question'].apply(clean_text)
    df['answer'] = df['answer'].apply(clean_text)
    df['hallucination'] = df['hallucination'].apply(clean_text)
    return df


def load_fever():
    dataset = load_dataset("fever", "fever")
    df = pd.DataFrame(dataset['train'])
    df['claim'] = df['claim'].apply(clean_text)
    df['evidence'] = df['evidence'].apply(lambda x: [clean_text(e) for e in x])
    df['label'] = df['label'].apply(clean_text)
    return df

def prepare_datasets():
    truthfulQA_ds = load_truthfulQA()
    haluEval_ds = load_HaluEval()
    fever_ds = load_fever()

    combined_ds = pd.concat([truthfulQA_ds, haluEval_ds, fever_ds], ignore_index=True)

    return combined_ds.to_dict(orient='records')


def evaluate_model_responses(model_responses):
    """
    Evaluate the model responses against the ground truths.
    This function should be customized based on the evaluation criteria.
    """
    results = []
    for response, truth in zip(model_responses):
        result = {
            "response": response,
            "ground_truth": truth,
            "is_correct": response == truth 
        }
        results.append(result)
    return results