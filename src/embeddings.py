from typing import List, Any
import numpy as np
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_community.embeddings import OllamaEmbeddings  # pip install langchain-community
from langchain_ollama import OllamaEmbeddings

from src.data_loader import load_all_documents


class EmbeddingPipeline:
    def __init__(self, model_name: str = "nomic-embed-text", chunk_size: int = 1000, chunk_overlap: int = 200, base_url: str = "http://localhost:11434"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Ollama must be running locally: `ollama serve`
        # And the model must be pulled: `ollama pull nomic-embed-text`
        self.model = OllamaEmbeddings(model=model_name, base_url=base_url)
        print(f"[INFO] Loaded Ollama embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks with Ollama...")

        # LangChain OllamaEmbeddings returns List[List[float]]
        vectors = self.model.embed_documents(texts)
        embeddings = np.array(vectors, dtype=np.float32)

        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings


if __name__ == "__main__":
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
