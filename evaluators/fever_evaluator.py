import json
from chain_of_thought import ChainOfThoughtLLM
from tqdm import tqdm
import random
from collections import defaultdict, Counter
from datasets import load_dataset, DownloadConfig
from config import HF_TOKEN
import os


class FEVEREvaluator:
    def __init__(self, model_name="gpt-4o-mini", inference_type="cot"):
        self.llm = ChainOfThoughtLLM(model_name=model_name)
        self.inference_type = inference_type

    def group_claims(self, data):
        grouped = defaultdict(lambda: {"claim": "", "labels": []})
        for i in range(len(data['id'])):
            claim_id = data['id'][i]
            claim = data['claim'][i]
            label = data['label'][i]

            grouped[claim_id]["claim"] = claim
            grouped[claim_id]["labels"].append(label)
        print("Grouped function ran")
        return grouped

    def majority_label(self, labels):
        count = Counter(labels)
        return count.most_common(1)[0][0]
    
    def evaluate(self, num_samples=100):
        data = load_dataset("fever", "v1.0", split="train", token=HF_TOKEN, trust_remote_code=True)
        sample_count = min(num_samples, len(data))
        print(f"Sample count: {sample_count}")
        print(f"Dataset Sample: {data[:20]}")

        
        data = data[:num_samples]
        correct = 0
        total = 0
        results = []

        # Convert FEVER-style labels to your expected format
        label_map = {
            "SUPPORTS": "Supported",
            "REFUTES": "Refuted",
            "NOT ENOUGH INFO": "NotEnoughInfo"
        }
        reverse_label_map = {v.lower(): k for k, v in label_map.items()}
        grouped_claims = self.group_claims(data)
        all_ids = list(grouped_claims.keys())[:num_samples]


        for cid in tqdm(all_ids):
            claim_data = grouped_claims[cid]
            claim = claim_data["claim"]
            label = self.majority_label(claim_data["labels"])

            prompt = f"""
                Claim: {claim}

                You are a fact verification assistant. Determine whether the above claim is:
                - Supported by facts (Supported)
                - Refuted by facts (Refuted)
                - Not enough info to verify (NotEnoughInfo)

                Respond with one of: Supported, Refuted, NotEnoughInfo.
                """
            
            cot = ChainOfThoughtLLM(model_name='gpt-4o-mini')
            # result = cot.answer_question(prompt, data_type="halueval")
            result = cot.answer_question(prompt, data_type="fever")
            predicted = result["final_answer"].strip().capitalize()

            # Normalize
            valid_labels = ["Supported", "Refuted", "Notenoughinfo"]
            if predicted not in valid_labels:
                predicted = "Invalid"

            mapped_predicted = reverse_label_map.get(predicted.lower(), "Invalid")
            is_correct = (mapped_predicted.lower() == label.lower())
            correct += int(is_correct)
            total += 1

            results.append({
                "claim_id": cid,
                "claim": claim,
                "ground_truth": label,
                "prediction": mapped_predicted,
                "is_correct": is_correct,
                "reasoning": result['full_reasoning']
            })

            print(results)

        accuracy = correct / total if total > 0 else 0
        print(f"FEVER Accuracy: {accuracy:.2f} ({correct}/{total})")

        return {
            "accuracy": accuracy,
            "individual_results": results
        }