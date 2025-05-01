import os
import collections
import re
from transformers import pipeline
from dotenv import load_dotenv
from huggingface_hub import login
from config import HF_TOKEN
from openai import OpenAI
import random
import json
import requests
import torch

# Load environment
load_dotenv()
login(token=os.getenv("HF_TOKEN"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SelfConsistencyLLM:
    def __init__(self, model_name="llama-2-7b", temperature=0.7, n=5):
        self.model_name = model_name
        self.temperature = random.uniform(0.3, 0.7) if None else temperature
        self.n = n
        self.client = openai_client if model_name == "gpt-3.5-turbo" else None
        self.pipeline = self._load_hf_pipeline() if model_name in ["llama-2-7b", "deepseek-r1"] else None

    def _load_hf_pipeline(self):
        if self.model_name == "llama-2-7b":
            return pipeline("text-generation", model="meta-llama/llama-2-7b-7b-chat-hf", device=0)
        elif self.model_name == "deepseek-r1":
            return pipeline("text-generation", model="deepseek-ai/deepseek-llm-7b-base", device=0)

    def _get_embedding(self, text):
        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()
    
    def extract_final_answer(self, text, data_type="halueval"):
        """Extracts the final answer string using regular expressions."""
        if data_type == "halueval":
            pattern = r"\s*(Yes|No)"
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        elif data_type == "fever":
            pattern = r"\s*(Supported|Refuted|NotEnoughInfo)"
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        patterns = [
            r"the answer is ([^\n\.]+)",
            r"the correct answer is ([^\n\.]+)",
            r"answer: ([^\n\.]+)",
            r"so, ([^\n\.]+) is the answer"
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1).strip()

        return text.strip().split('\n')[-1].strip()

    def generate_single_response(self, prompt):
        messages = [{"role": "user", "content": prompt}],
        max_length = 512
        temperature = random.uniform(0.3, 0.7)
        if self.model_name == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=512
                )
        elif self.model_name == "llama":
            pipe = pipeline(
                "text-generation",
                model="meta-llama/llama-2-7b-7b-chat-hf",
                token=HF_TOKEN
            )
            response = pipe(prompt, max_length=512, temperature=temperature)
            return response[0]["generated_text"]
        elif self.model_name == "deepseek":
        # time.sleep(10)
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv("DEEPSEEK_API_KEY")}",
                    "Content-Type": "application/json"
                },
                data=json.dumps({
                    "model": "deepseek/deepseek-r1-zero:free",
                    "messages": messages,
                    "temperature": temperature,
                    "max_length": max_length
                })
            )
            return response.choices[0].message.content.strip()
        else:
            output = self.pipeline(prompt, max_length=512, temperature=random.uniform(0.3, 0.7))
            return output[0]["generated_text"].strip()

    def generate_multiple_responses(self, prompt):
        responses = []
        for _ in range(self.n):
            try:
                response = self.generate_single_response(prompt)
                responses.append(response)
            except Exception as e:
                responses.append("ERROR")
        return responses

    def predict_consistency(self, answers):
        counter = collections.Counter(answers)
        return counter.most_common(1)[0][0]

    def answer_question(self, prompt, data_type=None):
        responses = self.generate_multiple_responses(prompt)
        final_answers = [self.extract_final_answer(resp, data_type) for resp in responses]
        majority_answer = self.predict_consistency(final_answers)

        return {
            "full_reasoning": responses,
            "final_answer": majority_answer,
        }
