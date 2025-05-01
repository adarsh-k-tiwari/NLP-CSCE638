import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from chain_of_thought import ChainOfThoughtLLM
from evaluators.truthfulqa_evaluator import TruthfulQAEvaluator
from evaluators.halueval import HaluEvalEvaluator
from evaluators.fever_evaluator import FEVEREvaluator
from deepeval.benchmarks.tasks import TruthfulQATask
from deepeval.benchmarks.modes import TruthfulQAMode
from config import HF_TOKEN, OPENAI_API_KEY

# Page configuration
st.set_page_config(
    page_title="LLM Chain of Thought Evaluation",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for storing results
if 'results' not in st.session_state:
    st.session_state.results = {
        'truthfulqa': None,
        'halueval': None,
        'fever': None
    }

# Title and description
st.title("LLM Chain of Thought Evaluation")
st.markdown("""
This application evaluates an LLM using various reasoning and verification techniques on three datasets:
- **TruthfulQA**: Assessing truthfulness in question answering
- **HaluEval**: Detecting hallucinations in generated content
- **FEVER**: Fact verification
""")

# Sidebar for model selection and configuration
st.sidebar.header("Model Configuration")

model_name = st.sidebar.selectbox(
    "Select LLM Model",
    ["gpt-3.5-turbo", "llama-2-7b", "deepseek-r1"]
)

model_type = st.sidebar.radio(
    "Select Inference Strategy",
    ["Base", "cot", "rag", "Self Consistency", "Self Verification"],
    index=1
)

# Initialize or load model
@st.cache_resource
def load_model(model_name):
    with st.spinner(f"Loading {model_name}..."):
        return ChainOfThoughtLLM(model_name, api_key=OPENAI_API_KEY)

model = load_model(model_name)

# Tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["Interactive Demo", "TruthfulQA Evaluation", "HaluEval Evaluation", "FEVER Evaluation"])

# Tab 1: Interactive Demo
with tab1:
    st.header("Test Reasoning Strategy")

    user_question = st.text_area("Enter a question:", "What is the capital of France?")

    if st.button("Generate Answer"):
        with st.spinner("Generating response..."):
            if model_type == "base":
                response = model.generate_response(user_question)
                st.subheader("Model Response (Base):")
                st.markdown(response)

            elif model_type == "cot":
                result = model.answer_question(user_question, None)
                st.subheader("Chain of Thought Reasoning:")
                st.markdown(result["full_reasoning"], unsafe_allow_html=True)
                st.subheader("Final Answer:")
                st.write(result["final_answer"])

            elif model_type == "rag":
                st.warning("RAG is not implemented yet.")

            elif model_type == "self consistency":
                st.warning("Self-consistency is not implemented yet.")

            elif model_type == "self verification":
                st.warning("Self-verification is not implemented yet.")

# Tab 2: TruthfulQA Evaluation
with tab2:
    st.header("TruthfulQA Benchmark Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        mode = st.radio(
            "Evaluation Mode",
            ["MC1 (Single Answer)", "MC2 (Multiple Answers)"],
            index=0
        )

        truthfulqa_mode = TruthfulQAMode.MC1 if mode == "MC1 (Single Answer)" else TruthfulQAMode.MC2

    with col2:
        sample_size = st.slider(
            "Sample Size",
            min_value=10,
            max_value=100,
            value=20,
            step=10
        )

    available_tasks = [task.name for task in TruthfulQATask]
    selected_tasks = st.multiselect(
        "Select Topics (optional)",
        available_tasks,
        []
    )

    task_enums = [getattr(TruthfulQATask, task) for task in selected_tasks] if selected_tasks else None

    if st.button("Run TruthfulQA Evaluation"):
        with st.spinner("Evaluating on TruthfulQA..."):
            evaluator = TruthfulQAEvaluator(model, mode=truthfulqa_mode)
            results = evaluator.evaluate(tasks=task_enums)
            st.session_state.results['truthfulqa'] = results

        st.success(f"Evaluation complete! Overall score: {results['overall_score']:.2f}")

        if results['individual_results']:
            df = pd.DataFrame(results['individual_results'])
            st.dataframe(df)
            
# Tab 3: HaluEval Evaluation
with tab3:
    st.header("HaluEval Hallucination Detection")

    sample_size = st.slider(
        "Sample Size",
        min_value=10,
        max_value=100,
        value=20,
        step=10,
        key="halueval_sample"
    )

    if st.button("Run HaluEval Evaluation"):
        with st.spinner("Evaluating on HaluEval..."):
            evaluator = HaluEvalEvaluator(model, model_type)
            results = evaluator.evaluate(num_samples=sample_size)
            st.session_state.results['halueval'] = results

        st.success(f"Evaluation complete! Hallucination rate: {results['hallucination_rate']:.2f}")

        if results['individual_results']:
            df = pd.DataFrame(results['individual_results'])
            st.dataframe(df)


# Tab 4: FEVER Evaluation
with tab4:
    st.header("FEVER Fact Verification")

    sample_size = st.slider(
        "Sample Size",
        min_value=10,
        max_value=100,
        value=20,
        step=10,
        key="fever_sample"
    )

    if st.button("Run FEVER Evaluation"):
        with st.spinner("Evaluating on FEVER..."):
            evaluator = FEVEREvaluator(model, model_type)
            results = evaluator.evaluate(num_samples=sample_size)
            st.session_state.results['fever'] = results

        st.success(f"Evaluation complete! Accuracy: {results['accuracy']:.2f}")

        if results['individual_results']:
            df = pd.DataFrame(results['individual_results'])
            st.dataframe(df)


# Add comparative analysis when all evaluations are complete
if all(st.session_state.results.values()):
    st.header("Comparative Analysis")
    summary = {
        "TruthfulQA Score": st.session_state.results['truthfulqa']['overall_score'],
        "HaluEval Hallucination Rate": st.session_state.results['halueval']['hallucination_rate'],
        "FEVER Accuracy": st.session_state.results['fever']['accuracy']
    }
    st.json(summary)
    fig, ax = plt.subplots()
    metrics = list(summary.keys())
    values = list(summary.values())
    ax.bar(metrics, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title('Performance Across Benchmarks')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)