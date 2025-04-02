import streamlit as st
from retrieval import simple_retrieval
from prompt_engineering import build_cot_prompt
from generation import generate_cot_paths
from self_consistency import select_consistent_response
from self_verification import verify_answer

st.title("CoT-RAG: Chain-of-Thought with Retrieval-Augmented Generation")
st.write("Enter a query to generate a response with step-by-step reasoning based on retrieved evidence.")

query = st.text_input("Enter your query:")

if st.button("Generate Response"):
    if not query:
        st.error("Please enter a valid query.")
    else:
        st.info("Retrieving evidence...")
        retrieved_docs = simple_retrieval(query)
        st.write("**Retrieved Evidence:**")
        for doc in retrieved_docs:
            st.markdown(f"**{doc['title']}**: {doc['content']}")
        
        st.info("Building Chain-of-Thought prompt...")
        prompt = build_cot_prompt(query, retrieved_docs)
        st.text_area("Prompt", prompt, height=150)
        
        st.info("Generating multiple reasoning paths...")
        cot_paths = generate_cot_paths(prompt, num_paths=5)
        st.write("**Chain-of-Thought Reasoning Paths:**")
        for idx, path in enumerate(cot_paths, start=1):
            st.markdown(f"**Path {idx}:**\n\n{path}")
        
        st.info("Selecting the most consistent answer...")
        selected_answer, final_answers = select_consistent_response(cot_paths)
        st.write("**Selected Answer:**")
        st.write(selected_answer)
        
        st.info("Verifying the answer...")
        verification = verify_answer(query, selected_answer, retrieved_docs)
        st.write("**Verification Feedback:**")
        st.write(verification)
