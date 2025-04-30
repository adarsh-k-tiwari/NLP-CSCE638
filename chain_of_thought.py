import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from openai import OpenAI
import re
from config import HF_TOKEN, OPENAI_API_KEY


class ChainOfThoughtLLM:
    def __init__(self, model_name="gpt-4o-mini", api_key=OPENAI_API_KEY):
        self.model_name = model_name
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

        # if model_name == "mistralai/Mistral-7B-v0.1":
        #     self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=HF_TOKEN)
        #     self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")
        #     self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0)
        
    def generate_cot_prompt(self, prompt):
        """Convert a question into a Chain of Thought prompt"""
        
        prompt = f"{prompt}\nLet's think step by step to solve this problem based on the context provided above:"
        return prompt
    
    def extract_final_answer(self, text, data_type="halueval"):
        # First, try to extract Yes/No after '#Your Judgement#:'
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
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_length
            )

            chunk = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            full_response += chunk

            # print(f"Chunk finish reason: {finish_reason}")

            if finish_reason != "length":
                break

            # For continuation: add context for the model
            messages = [
                {"role": "system", "content": "Continue the previous response."},
                {"role": "user", "content": full_response + "\nContinue."}
            ]

        return full_response
    
    def answer_question(self, prompt, data_type):
        """Complete pipeline to answer a question using CoT"""
        cot_prompt = self.generate_cot_prompt(prompt)
        print(f"Chain of Thought Prompt: {cot_prompt}")
        full_response = self.generate_response(cot_prompt)
        # response_no_cot = self.generate_response(question)
        
        # Extract the final answer after reasoning
        # This is a simple implementation - you might need more sophisticated parsing
        reasoning = full_response.replace(cot_prompt, "").strip()
        final_answer = self.extract_final_answer(reasoning, data_type)
        print(type(reasoning), type(final_answer))
        return {
            # "question": question,
            "full_reasoning": str(reasoning),
            # "response_no_cot": response_no_cot,
            "final_answer": str(final_answer)
        }
