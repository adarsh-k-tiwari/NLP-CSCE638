import json
import requests
from pinecone import Pinecone
from openai import OpenAI
from transformers import pipeline
import os
import time
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, classification_report
from dotenv import load_dotenv
load_dotenv()

from data_preprocessing import get_dataset

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(os.getenv("PINECONE_API_KEY"))

index_name = "truthful-qa"
index = pc.Index(index_name)

nltk.download('punkt_tab')


def get_embedding(text: str):
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding-Error: {e}")
        return None

def load_dict_from_json(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading dictionary: {e}")
        return None
    
def retrieve_documents(query_text: str, namespace: str = "ns1", top_k: int = 5):
    try:
        query_embedding = get_embedding(query_text)
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        return results
    except Exception as e:
        print(f"Retriever-Error: {e}")
        return None
    
def generate_response(query, 
                      model="openai", 
                      sys_prompt="", 
                      is_rag_enabled=False, 
                      max_tokens=100, 
                      dataset="TruthfulQA"):
    try:
        if is_rag_enabled:
            if dataset == "TruthfulQA":
                namespace = 'ns1'
                doc_store = load_dict_from_json("truthfulQA_documentStore.json")
            elif dataset == "HaluEval":
                namespace = 'ns2'
                doc_store = load_dict_from_json("haluEval_documentStore.json")
            else:
                namespace = 'ns3'
                doc_store = load_dict_from_json("fever_documentStore.json")

            docs = retrieve_documents(query, namespace)["matches"]
            context = ""
            for doc in docs:
                if dataset == "TruthfulQA":
                    context += "Domain: " + doc["metadata"]["domain"] + "\n"
                if dataset != "HaluEval":
                    context += "Title: " + doc["metadata"]["title"] + "\n"
                context += "Text: " + doc_store[doc["id"]] + "\n\n"
            sys_prompt = f"""{sys_prompt}
             
             Context:
             {context}"""

        if model == "openai":
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=max_tokens,
                temperature=0.5  
            )
            return response.choices[0].message.content.strip()
        
        elif model == "llama":
        
            pipe = pipeline(
                "text-generation",
                model="meta-llama/Llama-2-7b-chat-hf",
                token=os.getenv("HF_TOKEN")
            )
            prompt = f"""<s>[INST] <<SYS>>
            {sys_prompt}
            <</SYS>>

            {query} [/INST]"""
            response = pipe(prompt, max_length=512, temperature=0.5)
            return response[0]["generated_text"]
        
        elif model == "deepseek":
            time.sleep(10)
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv("DEEPSEEK_API_KEY")}",
                    "Content-Type": "application/json"
                },
                data=json.dumps({
                    "model": "deepseek/deepseek-r1-zero:free",
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": query}
                    ]
                })
            )

            try:
                generated_response = response.json()['choices'][0]['message']['content'].split('{')[1]
                if generated_response[0] == '"':
                    generated_response = generated_response[1:-2]
                else:
                    generated_response = generated_response[:-1]
                return generated_response
            except Exception as e:
                print(response.json()['choices'][0]['message']['content'])
                return ""
        
        else:
            raise ValueError(f"Unknown model: {model}")
        
    except Exception as e:
        print(f"Generate-Error: {e}")
        return None

def compute_mc2(generated, correct_answers, incorrect_answers):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    all_choices = correct_answers + incorrect_answers
    labels = [1] * len(correct_answers) + [0] * len(incorrect_answers)
    
    gen_emb = model.encode([generated])
    choice_embs = model.encode(all_choices)
    
    sims = cosine_similarity(gen_emb, choice_embs)[0]
    best_idx = int(np.argmax(sims))
    
    return labels[best_idx]

def evaluate_rag(dataset="TruthfulQA", model="openai"):
    print(f"Evaluating Dataset - {dataset}")
    try:
        df = get_dataset(dataset.lower()).iloc[:100]
        base_responses = []
        rag_responses = []

        if dataset == "TruthfulQA":
            mc2_base = []
            mc2_rag = []
            for i, row in df.iterrows():
                print(f"Processing {i+1} | {len(df)}")
                prompt = row['question']
                base_generated = generate_response(query=prompt,
                                                   model=model,
                                                   sys_prompt="Generate 1 sentence answer for the attached question.",
                                                   is_rag_enabled=False,
                                                   max_tokens=150)
                base_responses.append(base_generated)

                rag_generated = generate_response(query=prompt,
                                                   model=model,
                                                   sys_prompt="Generate 1 sentence answer for the attached question based on the following context.",
                                                   is_rag_enabled=True,
                                                   max_tokens=150,
                                                   dataset=dataset)
                rag_responses.append(rag_generated)

                mc2 = compute_mc2(base_generated, row['correct_answers'], row['incorrect_answers'])
                mc2_base.append(mc2)

                mc2 = compute_mc2(rag_generated, row['correct_answers'], row['incorrect_answers'])
                mc2_rag.append(mc2)
            

            print(f"\n\nMC2 (base): {sum(mc2_base) / len(mc2_base)} ")
            print()
            print(f"MC2 (rag): {sum(mc2_rag) / len(mc2_rag)}")
        
        elif dataset == "fever":
            # count = 0
            correct_base = 0
            correct_rag = 0
            actual_labels = []
            base_labels = []
            rag_labels = []
            for i, row in df.iterrows():
                label = row['label'].lower()
                if label == 'not enough info':
                    continue

                print(f"Processing {i+1} | {len(df)}")
                prompt = row['claim']
                base_generated = generate_response(query=prompt,
                                                   model=model,
                                                   sys_prompt="Generate 1 word answer either <supports> or <refutes>",
                                                   is_rag_enabled=False,
                                                   max_tokens=5)
                base_responses.append(base_generated)

                rag_generated = generate_response(query=prompt,
                                                   model=model,
                                                   sys_prompt="Generate 1 word answer either <supports> or <refutes> to the asked question based on the following context.",
                                                   is_rag_enabled=True,
                                                   max_tokens=5,
                                                   dataset=dataset)
                rag_responses.append(rag_generated)

                if label in base_generated:
                    correct_base += 1
                if label in rag_generated:
                    correct_rag += 1
                if 'supports' in label:
                    actual_labels.append(1)
                else:
                    actual_labels.append(0)
                if 'supports' in base_generated:
                    base_labels.append(1)
                else:
                    base_labels.append(0)
                if 'supports' in rag_generated:
                    rag_labels.append(1)
                else:
                    rag_labels.append(0)
            
            print("\n\nTotal Queries:", 500)
            print()
            print("Correct Predictions (Base):", correct_base)
            print("Accuracy (Base):", (correct_base/len(df))*100)
            print()
            print("Correct Predictions (RAG):", correct_rag)
            print("Accuracy (RAG):", (correct_rag/len(df))*100)
            print("\n\n")

            f1 = f1_score(actual_labels, base_labels, average='binary')
            print("Base Model\n")
            print(f"F1 Score - {f1}")
            print("\nClassification Report:")
            print(classification_report(actual_labels, base_labels))

            f1 = f1_score(actual_labels, rag_labels, average='binary')
            print("\nRAG Model\n")
            print(f"F1 Score - {f1}")
            print("\nClassification Report:")
            print(classification_report(actual_labels, rag_labels))

    except Exception as e:
        print(f"Evaluate-Error: {e}")
        return None
    

if __name__ == "__main__":

    print("------------------------------------------------------------\n")
    model = "openai"    
    print("Model:", model,"\n\n")

    dataset = "TruthfulQA"
    evaluate_rag(dataset=dataset, model=model)

    print("----------------------\n")
    dataset = "fever"
    evaluate_rag(dataset=dataset, model=model)

    print("------------------------------------------------------------\n")

    model = "llama"    
    print("Model:", model,"\n\n")

    dataset = "TruthfulQA"
    evaluate_rag(dataset=dataset, model=model)

    print("----------------------\n")
    dataset = "fever"
    evaluate_rag(dataset=dataset, model=model)

    print("------------------------------------------------------------\n")

    model = "deepseek"    
    print("Model:", model,"\n\n")

    dataset = "TruthfulQA"
    evaluate_rag(dataset=dataset, model=model)

    print("----------------------\n")
    dataset = "fever"
    evaluate_rag(dataset=dataset, model=model)

    print("------------------------------------------------------------")