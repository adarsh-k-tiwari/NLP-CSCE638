import unittest
from unittest.mock import patch
from self_verification import verify_answer  # Assuming the function is in the self_verification.py file

class TestVerifyAnswer(unittest.TestCase):
    @patch('generation.generate_response')  # Mocking the generate_response function
    def test_verify_answer(self, mock_generate_response):
        # Setup mock response from generate_response
        mock_generate_response.return_value = "Yes, the answer is correct based on the evidence provided."

        # Test with mock data
        query = "What is the capital of France?"
        selected_answer = "Paris"
        retrieved_docs = [
            {"title": "Document 1", "content": "France's capital is Paris."},
            {"title": "Document 2", "content": "Paris is the capital city of France."},
        ]
        
        response = verify_answer(query, selected_answer, retrieved_docs)
        
        # Check if the mock response was returned correctly
        self.assertEqual(response, "Yes, the answer is correct based on the evidence provided.")
        
        # Ensure generate_response was called once
        mock_generate_response.assert_called_once()  

if __name__ == '__main__':
    unittest.main()
