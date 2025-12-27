from src.data_loader import load_all_documents
from src.vector_store import FaissVectorStore
from src.search import RAGSearch

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    # store.build_from_documents(docs)
    # store.load()
    store.build_from_documents(docs)
    #print(store.query("What is attention mechanism?", top_k=3))
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)







# import os
# from src.data_loader import load_all_documents
# from src.vector_store import FaissVectorStore
# from src.search import RAGSearch

# if __name__ == "__main__":
#     persist_dir = "faiss_store"
#     faiss_path = os.path.join(persist_dir, "faiss.index")
#     meta_path = os.path.join(persist_dir, "metadata.pkl")

#     store = FaissVectorStore(
#         persist_dir=persist_dir,
#         embedding_model="nomic-embed-text",
#         base_url="http://localhost:11434",
#     )

#     # Build once if missing, otherwise load
#     if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
#         docs = load_all_documents("data")
#         store.build_from_documents(docs)
#     else:
#         store.load()

#     rag_search = RAGSearch(
#         persist_dir=persist_dir,
#         embedding_model="nomic-embed-text",
#         llm_model="gemma3:1b",   # lightweight
#         base_url="http://localhost:11434",
#     )

#     query = "What is attention mechanism?"
#     summary = rag_search.search_and_summarize(query, top_k=3)
#     print("Summary:", summary)


