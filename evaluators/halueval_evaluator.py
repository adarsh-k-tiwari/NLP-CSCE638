from config import HF_TOKEN
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from datasets import load_dataset, DownloadConfig
# from bluert import score

class HaluEvalEvaluator:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Embedding model
        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", token=HF_TOKEN)
        self.bert_model = AutoModel.from_pretrained("bert-base-uncased", token=HF_TOKEN).to(self.device)

        # NLI model
        self.nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli", token=HF_TOKEN)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli", token=HF_TOKEN).to(self.device)

        # BLEURT model
        self.bleurt_tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-512", token=HF_TOKEN)
        self.bleurt_model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512", token=HF_TOKEN).to(self.device)

    def evaluate(self, subset="QA", sample_size=100):
        samples = self._load_halueval_samples(subset, sample_size)
        
        results = []
        for sample in samples:
            question = sample["question"]
            knowledge = sample.get("knowledge", "")

            # Format input
            if subset == "QA":
                input_text = f"Question: {question}\nContext: {knowledge}"
            elif subset == "Dialogue":
                input_text = f"Background: {knowledge}\nConversation: {question}. Please give me the next dialogue for [Assistant]."
            elif subset == "Summarization":
                input_text = f"Document to summarize: {knowledge}. Please summarize this for me."
            else:
                input_text = question

            response = self.model.answer_question(input_text)

            results.append({
                "input": input_text,
                "model_reasoning": response["full_reasoning"],
                "model_answer": response["final_answer"],
                "contains_hallucination": self._detect_hallucination(response["final_answer"], sample)
            })

        hallucination_rate = sum(r["contains_hallucination"] for r in results) / len(results) if results else 0
        
        return {
            "hallucination_rate": hallucination_rate,
            "individual_results": results
        }

    def _load_halueval_samples(self, subset, sample_size):
        try:
            subset_map = ["qa", "dialogue", "summarization", "general"]
            config_name = subset.lower()
            if config_name not in subset_map:
                raise ValueError(f"Invalid subset: {subset}. Valid options: {subset_map}")
            download_config = DownloadConfig(token=HF_TOKEN)

            dataset = load_dataset("pminervini/HaluEval", name=config_name, split='data', token=HF_TOKEN)
            sample_count = min(sample_size, len(dataset))
            samples = dataset.select(range(sample_count))

            return [
                {
                    "question": sample.get("question", sample.get("dialogue_history", "")),
                    "knowledge": sample.get("knowledge", sample.get("document", "")),
                    "answer": sample.get("right_answer", sample.get("right_summary", sample.get("right_response", ""))),
                    "hallucinated_answer": sample.get("hallucinated_answer", sample.get("hallucinated_response", sample.get("hallucinated_summary", ""))),
                }
                for sample in samples
            ]
        except Exception as e:
            print(f"Error loading dataset from Hugging Face: {e}")
            return []

    def _get_embedding(self, text):
        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    def _contradiction_score(self, premise, hypothesis):
        encoded = self.nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.nli_model(**encoded).logits
            probs = F.softmax(logits, dim=-1).squeeze()
        return probs[2].item()

    def _compute_bleurt_score(self, candidate, reference):
        inputs = self.bleurt_tokenizer(reference, candidate, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            scores = self.bleurt_model(**inputs).logits
        return scores.squeeze().item()

    def _detect_hallucination(self, response, sample):
        ground_truth = sample.get("answer", "").strip().lower()
        hallucinated_answer = sample.get("hallucinated_answer", "").strip().lower()
        model_answer = response.strip().lower()

        if model_answer == ground_truth or ground_truth in model_answer:
            return False

        gt_embedding = self._get_embedding(ground_truth)
        resp_embedding = self._get_embedding(model_answer)
        hallucinated_embedding = self._get_embedding(hallucinated_answer)

        sim_score = F.cosine_similarity(gt_embedding.unsqueeze(0), resp_embedding.unsqueeze(0)).item()
        hallucinated_score = F.cosine_similarity(hallucinated_embedding.unsqueeze(0), resp_embedding.unsqueeze(0)).item()
        
        return any([
            sim_score < 0.55,
            hallucinated_score > 0.4
        ])
