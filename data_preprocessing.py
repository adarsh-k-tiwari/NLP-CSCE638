import pandas as pd
import json
import os
from datasets import load_dataset


def clean_text(text):
    if isinstance(text, str):
        text = text.lower().strip()
        cleaned_text = text.replace('\n', ' ').replace('\r', '').strip()
        return cleaned_text
    elif isinstance(text, list):
        cleaned_texts = []
        for txt in text:
            txt = txt.lower().strip()
            txt = txt.replace('\n', ' ').replace('\r', '').strip()
            cleaned_texts.append(txt)
        return cleaned_texts
    return None


def load_truthfulQA():
    dataset = load_dataset("truthfulqa/truthful_qa", "generation")
    df = pd.DataFrame(dataset['validation'])
    df['question'] = df['question'].apply(clean_text)
    df['source'] = df['source'].apply(clean_text)
    df['best_answer'] = df['best_answer'].apply(clean_text)
    df['correct_answers'] = df['correct_answers'].apply(clean_text)
    df['incorrect_answers'] = df['incorrect_answers'].apply(clean_text)
    return df


def load_HaluEval():
    dataset = load_dataset("pminervini/HaluEval", "qa")
    df = pd.DataFrame(dataset['data'])
    df['knowledge'] = df['knowledge'].apply(clean_text)
    df['question'] = df['question'].apply(clean_text)
    df['answer'] = df['answer'].apply(clean_text)
    df['hallucination'] = df['hallucination'].apply(clean_text)
    return df


def load_fever():
    dataset = load_dataset("fever", "v1.0")
    df = pd.DataFrame(dataset['train'])
    df['id'] = df['id']
    df['claim'] = df['claim'].apply(clean_text)
    df['evidence_id'] = df['evidence_id']
    df['evidence_wiki_url'] = df['evidence_wiki_url'].apply(clean_text)
    df['label'] = df['label'].apply(clean_text)
    df['evidence_sentence_id'] = df['evidence_sentence_id']
    df['evidence_annotation_id'] = df['evidence_annotation_id']
    return df


def prepare_datasets():
    truthfulQA_ds = load_truthfulQA()
    haluEval_ds = load_HaluEval()
    fever_ds = load_fever()

    combined_ds = pd.concat([truthfulQA_ds, haluEval_ds, fever_ds], ignore_index=True)

    return combined_ds.to_dict(orient='records')


def get_dataset(type="truthfulqa"):
    try:
        if type == "truthfulqa":
            df = load_truthfulQA()
            df = df.sample(n=500, random_state=42)
        elif type == "halueval":
            df = load_HaluEval()
            df = df.sample(n=500, random_state=42)
        else:
            df = load_fever()
            df = df[df['label'].isin(['supports', 'refutes'])].sample(n=500, random_state=42)
        
        return df
    
    except Exception as e:
        print(f"LoadData-Error: {e}")
        return None