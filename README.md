# LLM Evaluation Dashboard – CSCE 638 Project

This repository provides a Streamlit web application for evaluating hallucinations and factual accuracy in Large Language Models (LLMs) using multiple techniques including **Chain-of-Thought (CoT)**, **Retrieval-Augmented Generation (RAG)**, **Self-Consistency**, and **Self-Verification** across datasets like **TruthfulQA**, **FEVER**, and **HaluEval**.

---

## Features

- ✅ Evaluate LLMs on multiple datasets for factual correctness.
- 🧠 Chain-of-Thought reasoning generation.
- 📚 Retrieval-Augmented Generation with Pinecone-based document store.
- 🔁 Self-Consistency decoding with multiple generations.
- 🕵️ Self-Verification: use one model to critique another.
- 📊 Visualize performance via Streamlit interface.

---

## Project Structure
```
.NLP-CSCE638/
|-- app.py # Streamlit app entry point 
|-- config.py # Configuration (API keys, settings) 
|-- chain_of_thought.py # CoT model logic 
|-- rag.py # RAG implementation 
|-- self_consistency.py # Self-consistency evaluator 
|-- self_verification.py # Self-verification evaluator 
|-- data_preprocessing.py # Dataset loaders and cleaning 
|-- evaluators/ 
|  |-- fever_evaluator.py 
|  |-- halueval_evaluator.py 
|  |-- truthfulqa_evaluator.py 
|-- results/ # Evaluation outputs 
|-- requirements.txt # Python dependencies
```

---

## Getting Started

### 1. Clone the Repository


```git clone https://github.com/adarsh-k-tiwari/NLP-CSCE638.git```

```cd NLP-CSCE638```

### 2. Install Requirements

We recommend using a virtual environment:

```pip install -r requirements.txt```


### 3. Set up Environment Variables

- Create `.env` files in root directory
- Define ```OPENAI_API_KEY, PINECONE_API_KEY, HF_TOKEN, DEEPSEEK_API_KEY```

### 4. Run the App
```streamlit run app.py```


### 5. Supported Datasets
- **TruthfulQA** - The TruthfulQA (generation) dataset is a benchmark designed to evaluate the ability of language models to generate factually correct and non-misleading answers, particularly in the presence of common misconceptions or false beliefs. It consists of 817 samples, each containing a question, a list of correct answers, and a list of incorrect (but often plausible-sounding) answers. The questions cover a wide range of open-ended topics, making the dataset suitable for assessing truthfulness and robustness in language models.
- **FEVER** - FEVER (v1.0) (Thorne et al., 2018) is a large-scale benchmark dataset developed to evaluate a model’s ability to verify factual claims using evidence from a structured knowledge base (Wikipedia). It is widely used in fact-checking and evidence-based reasoning tasks. It consists of a label indicating whether the claim is Supported, Refuted, Not Enough Info. It has approximately 145,000 claim evidence pairs
- **HaluEval** - HaluEval (qa) (Li et al., 2023) is a benchmark dataset designed to evaluate hallucination in large language model outputs. It includes questions or prompts paired with correct response, halucinated response and knowledge to support the correct answer. The dataset that we are using contains 4,079 samples in total, focusing on Open-domain QA.

### 6. Supported Evaluation Modes
Base Model: Direct answer from LLM

CoT: Chain-of-Thought reasoning

RAG: LLM + Pinecone document retrieval

Self-Consistency: Majority voting over multiple completions

Self-Verification: LLM critic judging factuality

### 7. License
This project is licensed under the MIT License. See the `LICENSE` file for details.

### 8. Acknowledgements
This is a course project for CSCE 638: Natural Language Processing at Texas A&M University.