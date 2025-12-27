import os
from src.vector_store import FaissVectorStore
# from langchain_community.chat_models import ChatOllama  
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama




class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "nomic-embed-text",  
        llm_model: str = "gemma3:1b",                  
        base_url: str = "http://localhost:11434",
    ):
        # Vectorstore (retrieval)
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents  # fixed import
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        # LLM (generation) via Ollama
        # Requirements:
        #   ollama serve
        #   ollama pull llama3
        self.llm = ChatOllama(model=llm_model, base_url=base_url)
        print(f"[INFO] Ollama LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)

        texts = [r["metadata"].get("text", "") for r in results if r.get("metadata")]
        context = "\n\n".join([t for t in texts if t.strip()])

        if not context:
            return "No relevant documents found."

        prompt = (
            f"You are a helpful assistant. Use ONLY the context to answer.\n\n"
            f"Query: {query}\n\n"
            f"Context:\n{context}\n\n"
            f"Answer:"
        )

        
        response = self.llm.invoke(prompt)
        return response.content


if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
