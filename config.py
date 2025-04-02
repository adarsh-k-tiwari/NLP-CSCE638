# LLM API configuration (example with OpenAI)
OPENAI_API_KEY = "YOUR_KEY"
LLM_MODEL = "gpt-4o-mini"

# Retrieval configuration
DOCUMENTS = [
    {
        "id": 1,
        "title": "Document 1",
        "content": "This document contains facts about topic A..."
    },
    {
        "id": 2,
        "title": "Document 2",
        "content": "This document explains the background on topic B..."
    }
]

# Other parameters
NUM_COT_PATHS = 5  # Number of reasoning paths for self-consistency
