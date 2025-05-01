import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from openai import OpenAI
import re
from config import HF_TOKEN, OPENAI_API_KEY
import json
import requests
import os

class ChainOfThoughtLLM:
    def __init__(self, model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY):
        self.model_name = model_name
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        
    def generate_cot_prompt(self, prompt):
        """Convert a question into a Chain of Thought prompt"""
        
        prompt = f"{prompt}\nLet's think step by step to solve this problem based on the context provided above:"
        return prompt
    
    def extract_final_answer(self, text, data_type=None):
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
            r"so, ([^\n\.]+) is the answer",
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1).strip()
        return text.strip().split('\n')[-1].strip()
    
    def generate_response(self, prompt, max_length=5000, temperature=0.2):
        """Generate a response using Chain of Thought reasoning"""
        full_response = ""
        messages = [{"role":"system",
                         "content": "You are a factually correct assistant. Think step by step, and always base your answer on facts present in the context."},
                       {"role": "user",
                       "content": prompt}]

        while True:
            if self.model_name == "gpt-3.5-turbo":
                response = self.client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_length
                )
                full_response = response.choices[0].message.content

            elif self.model_name == "llama-2-7b":
                pipe = pipeline(
                    "text-generation",
                    model="meta-llama/Llama-2-7b-chat-hf",
                    token=HF_TOKEN
                )
                response = pipe(messages, max_length=512, temperature=0.5)
                full_response = response.choices[0].message.content

            elif self.model_name == "deepseek-r1":
            # time.sleep(10)
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.getenv("DEEPSEEK_API_KEY")}",
                        "Content-Type": "application/json"
                    },
                    data=json.dumps({
                        "model": "deepseek/deepseek-r1-zero:free",
                        "messages": messages
                    })
                )
                full_response = response.choices[0].message.content
        return full_response

    def answer_question(self, prompt, data_type=None):
        """Complete pipeline to answer a question using CoT"""
        cot_prompt = self.generate_cot_prompt(prompt)
        
        full_response = self.generate_response(cot_prompt)
        
        # Extract the final answer after reasoning
        reasoning = full_response.replace(cot_prompt, "").strip()
        final_answer = self.extract_final_answer(reasoning, data_type)
        
        return {
            "full_reasoning": str(reasoning),
            "final_answer": str(final_answer)
        }
