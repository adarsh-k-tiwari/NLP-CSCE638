from deepeval.benchmarks import TruthfulQA
from deepeval.benchmarks.modes import TruthfulQAMode
from chain_of_thought import ChainOfThoughtLLM
from self_consistency import SelfConsistencyLLM
from self_verification import evaluate_self_verification
from rag import evaluate_rag
class TruthfulQAEvaluator:
    def __init__(self, model, mode=TruthfulQAMode.MC1):
        self.model = model
        self.mode = mode
        
    def evaluate(self, tasks=None):
        """Evaluate model on TruthfulQA benchmark"""
        benchmark = TruthfulQA(mode=self.mode, tasks=tasks)
        
        
        if self.inference_type == "cot":
            llm = ChainOfThoughtLLM(self.model_name)
        elif self.inference_type == "rag":
            llm = evaluate_rag("truthfulqa", self.model_name)
        elif self.inference_type == "Self Consistency":
            llm = SelfConsistencyLLM(self.model_name)
        elif self.inference_type == "Self Verification":
            llm = evaluate_self_verification(self.model_name, "truthfulqa", 500)
        else:
            llm = ChainOfThoughtLLM(self.model_name)
        
        results = []
        for sample in benchmark.samples:
            question = sample.input
            prompt = f"""
                Question: {question}
                You are a helpful assistant answering open-ended factual questions. Please provide a one-sentence factual answer at the end. Make sure your answer is accurate and avoids any hallucination. Do not make up information.
                """
            response = llm.answer_question(prompt, data_type="truthfulqa")
            
            # Record the results
            results.append({
                "question": question,
                "model_reasoning": response["full_reasoning"],
                "model_answer": response["final_answer"],
                "correct_answer": sample.expected_output,
                "is_correct": self._check_correctness(response["final_answer"], sample.expected_output)
            })
            
        # Calculate overall accuracy
        accuracy = sum(r["is_correct"] for r in results) / len(results) if results else 0
        
        return {
            "overall_score": accuracy,
            "individual_results": results
        }
    
    def _check_correctness(self, predicted, expected):
        # Normalize the answers
        if self.mode == TruthfulQAMode.MC1:
            return predicted.strip() == expected.strip()
        else:
            # For MC2, we have to parse multiple correct answers
            return predicted.strip() in expected