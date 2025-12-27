import streamlit as st
from src.search import RAGSearch

@st.cache_resource
def load_rag():
    return RAGSearch(
        persist_dir="faiss_store",
        embedding_model="nomic-embed-text",
        llm_model="gemma3:1b",
        base_url="http://localhost:11434",
    )

def main():
    st.set_page_config(page_title="Local RAG with Ollama + FAISS", layout="wide")
    st.title("Local RAG with Ollama + FAISS")

    rag = load_rag()

    query = st.text_input("Ask a question")
    if st.button("Ask") and query.strip():
        with st.spinner("Thinking..."):
            answer = rag.search_and_summarize(query, top_k=1)

        st.subheader("Answer")
        st.write(answer)

if __name__ == "__main__":
    main()
