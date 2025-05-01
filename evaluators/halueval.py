import json
from chain_of_thought import ChainOfThoughtLLM
from self_consistency import SelfConsistencyLLM
from self_verification import evaluate_self_verification
from rag import evaluate_rag
from config import HF_TOKEN
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random

class HaluEvalEvaluator:
    def __init__(self, model_name="gpt-3.5-turbo", inference_type="cot"):
        self.inference_type = inference_type
        # Embedding model
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", token=HF_TOKEN)
        self.bert_model = AutoModel.from_pretrained("bert-base-uncased", token=HF_TOKEN).to(self.device)

    def _get_embedding(self, text):
        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()
    
    def evaluate(self, num_samples=100):
        data_file = "evaluators/halueval.json"
        with open(data_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

        data = data[:num_samples]
        correct = 0
        total = 0
        individual_results = []

        for item in tqdm(data):
            question = item['question']
            knowledge = item.get('knowledge', '')
            right_answers = item.get('right_answer', "")
            hallucinated_answers = item.get('hallucinated_answer', "")

            if isinstance(right_answers, str):
                right_answers = [right_answers]
            if isinstance(hallucinated_answers, str):
                hallucinated_answers = [hallucinated_answers]

            all_answers = [(ans, 'No') for ans in right_answers] + [(ans, 'Yes') for ans in hallucinated_answers]

            if not all_answers:
                continue

            answer, label = random.choice(all_answers)
            if self.inference_type == "rag":
                prompt = f"Context: {knowledge}\nQuestion: {question}\nAnswer: {answer}\n"
            elif self.inference_type == "cot" or self.inference_type == "Self Consistency":
                prompt = f"Question: {question}\nAnswer: {answer}\n"
            else:
                prompt = f"Question: {question}\nAnswer: {answer}\n"

            prompt += """I want you act as an answer judge. Given a question and an answer, your objective is to determine if the provided answer contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.
                    You are trying to determine if the answer misunderstands the question context and intention.
                    #Question#: What is a rare breed of dog that was derived as a variant of Rat Terrier, Shiloh Shepherd dog or American Hairless Terrier?
                    #Answer#: American Hairless Terrier
                    #Your Judgement#: No

                    You are trying to determine if there is a factual contradiction between the answer and the world knowledge. Some information in the answer might be fabricated.
                    #Question#: Are the New Orleans Outfall Canals the same length as the Augusta Canal?
                    #Answer#: No, the New Orleans Outfall Canals and the Augusta Canal are not the same length. The Orleans Canal is approximately 3.6 miles (5.8 kilometers) long while the Augusta Canal is approximately 7 miles (11.3 kilometers) long.
                    #Your Judgement#: Yes
                    #Question#: What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?
                    #Answer#: U.S Highway 70
                    #Your Judgement#: Yes

                    You are trying to determine if the answer is too general or too specific to answer the question at an appropriate level of specificity.
                    #Question#: What genre do Superheaven and Oceansize belong to?
                    #Answer#: Superheaven and Oceansize belong to the rock genre.
                    #Your Judgement#: No
                    #Question#: What profession do Kōbō Abe and Agatha Christie share?
                    #Answer#: Playwright.
                    #Your Judgement#: No

                    You are trying to determine if the answer can be correctly inferred from the knowledge.
                    #Question#: Which band has more members, Muse or The Raconteurs?
                    #Answer#: Muse has more members than The Raconteurs.
                    #Your Judgement#: Yes
                    #Question#: Which is currently more valuable, Temagami-Lorrain Mine or Meadowbank Gold Mine?
                    #Answer#: Meadowbank Gold Mine, since Meadowbank Gold Mine is still producing gold and the TemagamiLorrain Mine has been inactive for years.
                    #Your Judgement#: No

                    You should try your best to determine if the answer contains non-factual or hallucinated information according to the above hallucination types. Do not add any extra words in your judgement or final answer. The answer you give MUST be either \"Yes\" or \"No\"."""

            if self.inference_type == "cot":
                llm = ChainOfThoughtLLM(self.model_name)
            elif self.inference_type == "rag":
                llm = evaluate_rag("fever", self.model_name)
            elif self.inference_type == "Self Consistency":
                llm = SelfConsistencyLLM(self.model_name)
            elif self.inference_type == "Self Verification":
                llm = evaluate_self_verification(self.model_name, "fever", num_samples)
            
            result = llm.answer_question(prompt, data_type="halueval")
            predicted = result['final_answer'].strip().capitalize()
            gt_embedding = self._get_embedding(right_answers)
            resp_embedding = self._get_embedding(predicted)

            sim_score = F.cosine_similarity(gt_embedding.unsqueeze(0), resp_embedding.unsqueeze(0)).item()
            if predicted not in ['Yes', 'No']:
                predicted = 'Invalid'

            is_correct = (predicted == label)
            correct += int(is_correct)
            total += 1

            individual_results.append({
                "question": question,
                "knowledge": knowledge,
                "answer": answer,
                "ground_truth": label,
                "judgement": predicted,
                "is_correct": is_correct,
                "full_reasoning": result['full_reasoning'],
                "final_answer": result['final_answer']
            })

        accuracy = correct / total if total > 0 else 0
        hallucination_rate = 1 - accuracy if total > 0 else 0

        # print(f"Accuracy: {accuracy:.2f} ({correct}/{total})")

        return {
            "accuracy": accuracy,
            "hallucination_rate": hallucination_rate,
            "individual_results": individual_results
        }
