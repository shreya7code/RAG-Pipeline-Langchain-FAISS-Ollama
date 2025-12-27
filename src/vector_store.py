import os
import faiss
import numpy as np
import pickle
from typing import List, Any

# from langchain_community.embeddings import OllamaEmbeddings  
from langchain_ollama import OllamaEmbeddings

from src.embeddings import EmbeddingPipeline


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        base_url: str = "http://localhost:11434",
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []

        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Ollama embedding model for query-time embeddings
        # Requirements:
        #   ollama serve
        #   ollama pull nomic-embed-text
        self.embedder = OllamaEmbeddings(model=embedding_model, base_url=base_url)

        print(f"[INFO] Loaded Ollama embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")

        # Uses your EmbeddingPipeline which chunks + embeds documents using OllamaEmbeddings
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)  # np.ndarray float32 already

        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings, dtype=np.float32), metadatas)

        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D (n, dim). Got shape: {embeddings.shape}")

        dim = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        else:
            # Safety: ensure new vectors match existing index dimension
            if self.index.d != dim:
                raise ValueError(
                    f"Embedding dim mismatch. Index dim={self.index.d} but embeddings dim={dim}. "
                    f"Did you rebuild the index with a different embedding model?"
                )

        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)

        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        if self.index is None:
            raise ValueError("Cannot save: FAISS index is empty. Build or add embeddings first.")

        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(
                f"Missing FAISS artifacts in {self.persist_dir}. Expected faiss.index and metadata.pkl"
            )

        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index is None:
            raise ValueError("FAISS index not loaded. Call load() or build_from_documents() first.")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Ensure float32 for FAISS
        query_embedding = query_embedding.astype(np.float32)

        # Safety: ensure query dim matches index dim
        if query_embedding.shape[1] != self.index.d:
            raise ValueError(
                f"Query embedding dim={query_embedding.shape[1]} does not match index dim={self.index.d}. "
                f"Are you using the same embedding model used to build the index?"
            )

        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if 0 <= idx < len(self.metadata) else None
            results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")

        # Using Ollama embedder for query embedding 
        vec = self.embedder.embed_query(query_text)  # List[float]
        query_emb = np.array(vec, dtype=np.float32).reshape(1, -1)

        return self.search(query_emb, top_k=top_k)


if __name__ == "__main__":
    from src.data_loader import load_all_documents

    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store", embedding_model="nomic-embed-text")
    store.build_from_documents(docs)

    store.load()
    print(store.query("What is attention mechanism?", top_k=3))
