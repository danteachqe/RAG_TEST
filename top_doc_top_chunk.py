from typing import List, Dict
import os
import faiss
import numpy as np

# Config
DOCS_DIR = "./Documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_DIM = 384

# -------- Utilities --------

def extract_text_from_pdf(filepath: str) -> str:
    # Simulate PDF content
    return f"Simulated content of PDF: {os.path.basename(filepath)}." * 10

def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

def embed_texts(texts: List[str]) -> np.ndarray:
    return np.random.rand(len(texts), EMBEDDING_DIM).astype("float32")

# -------- Main RAG Function --------

def retrieve_best_doc_and_top_chunks(query: str) -> Dict:
    doc_vectors = []
    doc_names = []
    doc_chunks_map = {}

    # Step 1: Compute average vector for each doc
    for filename in os.listdir(DOCS_DIR):
        if not (filename.endswith(".pdf") or filename.endswith(".txt")):
            continue

        path = os.path.join(DOCS_DIR, filename)
        try:
            if filename.endswith(".pdf"):
                text = extract_text_from_pdf(path)
            else:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        doc_chunks_map[filename] = chunks
        embeddings = embed_texts(chunks)
        avg_vec = np.mean(embeddings, axis=0, keepdims=True)
        doc_vectors.append(avg_vec)
        doc_names.append(filename)

    if not doc_vectors:
        return {"error": "No valid documents found."}

    # Step 2: Retrieve best-matching document
    doc_matrix = np.vstack(doc_vectors)
    query_vec = embed_texts([query])

    faiss.normalize_L2(doc_matrix)
    faiss.normalize_L2(query_vec)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(doc_matrix)
    _, I = index.search(query_vec, 1)
    best_doc_idx = I[0][0]
    best_doc = doc_names[best_doc_idx]

    # Step 3: Score top 3 chunks from best doc
    chunks = doc_chunks_map[best_doc]
    chunk_embeddings = embed_texts(chunks)

    faiss.normalize_L2(chunk_embeddings)
    faiss.normalize_L2(query_vec)

    chunk_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    chunk_index.add(chunk_embeddings)
    D, I = chunk_index.search(query_vec, 3)

    top_chunks = [{
        "chunk_text": chunks[i],
        "chunk_score": float(D[0][rank])
    } for rank, i in enumerate(I[0])]

    return {
        "document": best_doc,
        "top_chunks": top_chunks
    }

# --------- Test ---------

if __name__ == "__main__":
    query = "How does retrieval-augmented generation work?"
    result = retrieve_best_doc_and_top_chunks(query)

    print(f"\n📄 Best Document: {result['document']}")
    for i, chunk in enumerate(result['top_chunks'], 1):
        print(f"\n🔹 Chunk {i} (Score: {chunk['chunk_score']:.4f}):\n{chunk['chunk_text'][:200]}...")
