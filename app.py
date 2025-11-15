import streamlit as st
from rag_pipeline import answer_query, build_vector_db
import os

st.title("📘 Free PDF RAG Assistant")
st.write("100% free RAG system using local embeddings + Groq LLM.")

if st.sidebar.button("Rebuild Vector DB"):
    with st.spinner("Building vector DB..."):
        build_vector_db()
    st.success("Done!")

query = st.text_input("PDF se kuchh bhi puchho")

if st.button("Ask"):
    if query:
        with st.spinner("Thinking..."):
            answer = answer_query(query)
        st.write("### Answer")
        st.write(answer)
