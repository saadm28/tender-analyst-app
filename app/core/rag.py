# app/core/rag.py
from typing import Callable, Dict, List, Tuple
import numpy as np
import faiss

def build_faiss(chunks: List[str], embed_fn: Callable[[List[str]], np.ndarray], batch_size: int = 128
               ) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    """Build a cosine-similarity FAISS index from text chunks (batched, float32, L2-normalized)."""
    if not chunks:
        return None, None

    vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        arr = np.asarray(embed_fn(batch), dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        vectors.append(arr)

    embs = np.vstack(vectors)
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index, embs

def retrieve(query: str, chunks: List[str], index, embed_fn: Callable[[List[str]], np.ndarray], k: int = 8
            ) -> Dict:
    """Return the top-k chunks with separators, ready for LLM context."""
    if not chunks or index is None:
        return {"context": "", "indices": [], "snippets": [], "scores": []}
    q = np.asarray(embed_fn([query]), dtype="float32")
    if q.ndim == 1:
        q = q.reshape(1, -1)
    faiss.normalize_L2(q)
    k = min(k, len(chunks))
    scores, idx = index.search(q, k)
    ids = idx[0].tolist()
    snips = [chunks[i] for i in ids]
    return {
        "context": "\n\n---\n\n".join(snips),
        "indices": ids,
        "snippets": snips,
        "scores": scores[0].tolist(),
    }


















# import numpy as np
# import faiss
# from typing import Callable, Dict, List


# def build_faiss(chunks: list[str], embed_fn: Callable) -> tuple:
#     """Build FAISS index from text chunks."""
#     if not chunks:
#         return None, None
    
#     # Generate embeddings in batches to handle memory efficiently
#     batch_size = 50
#     embeddings = []
    
#     for i in range(0, len(chunks), batch_size):
#         batch = chunks[i:i + batch_size]
#         batch_embeddings = embed_fn(batch)
#         embeddings.append(batch_embeddings)
    
#     # Combine all embeddings
#     embeddings = np.vstack(embeddings)
    
#     # L2 normalize for cosine similarity
#     faiss.normalize_L2(embeddings)
    
#     # Create FAISS index
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
#     index.add(embeddings)
    
#     return index, embeddings


# def retrieve(query: str, chunks: list[str], index, embed_fn: Callable, k: int = 6) -> dict:
#     """Retrieve relevant chunks using FAISS similarity search."""
#     if not chunks or index is None:
#         return {"context": "", "indices": [], "snippets": []}
    
#     # Generate query embedding
#     query_embedding = embed_fn([query])
#     faiss.normalize_L2(query_embedding)
    
#     # Search for similar chunks
#     k = min(k, len(chunks))  # Don't search for more chunks than available
#     scores, indices = index.search(query_embedding, k)
    
#     # Get retrieved chunks
#     retrieved_chunks = [chunks[idx] for idx in indices[0]]
    
#     # Create context by joining retrieved chunks
#     context = "\n\n".join(retrieved_chunks)
    
#     return {
#         "context": context,
#         "indices": indices[0].tolist(),
#         "snippets": retrieved_chunks
#     }
