from deepeval.benchmarks import TruthfulQA
from deepeval.benchmarks.modes import TruthfulQAMode

class TruthfulQAEvaluator:
    def __init__(self, model, mode=TruthfulQAMode.MC1):
        self.model = model
        self.mode = mode
        
    def evaluate(self, tasks=None):
        """Evaluate model on TruthfulQA benchmark"""
        benchmark = TruthfulQA(mode=self.mode, tasks=tasks)
        
        # Custom evaluation to use our CoT model
        results = []
        for sample in benchmark.samples:
            question = sample.input
            response = self.model.answer_question(question)
            
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
        """Simple correctness check - customize based on TruthfulQA mode"""
        # This is a simplified implementation
        if self.mode == TruthfulQAMode.MC1:
            return predicted.strip() == expected.strip()
        else:
            # For MC2, we'd need to parse multiple correct answers
            return predicted.strip() in expected
