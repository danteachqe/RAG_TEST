# Simple Retriever using FAISS and OpenAI Embeddings
import os, pickle, sys
from typing import List, Dict
import numpy as np
import fitz                         # PyMuPDF
import faiss                       # Facebook AI Similarity Search
from tqdm import tqdm
import openai

# -------------------------- Config ---------------------------------
DB_DIR      = "faiss_index"
INDEX_FILE  = os.path.join(DB_DIR, "index.faiss")
META_FILE   = os.path.join(DB_DIR, "docs.pkl")
CHUNK_SIZE  = 500                 # characters
CHUNK_OVERLAP = 100
EMB_MODEL   = "text-embedding-3-small"  # or any OpenAI embedding model
# -------------------------------------------------------------------

# ---------- 1. PDF/TXT LOADING ------------------------------------
def extract_text_from_pdf(path: str) -> str:
    text = []
    with fitz.open(path) as doc:
        for page in doc:
            text.append(page.get_text("text"))
    return "\n".join(text)

def load_documents(folder: str) -> List[Dict]:
    """Return [{text: str, metadata: dict}, ...]"""
    docs = []
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if fname.lower().endswith(".pdf"):
            raw = extract_text_from_pdf(fpath)
        elif fname.lower().endswith(".txt"):
            raw = open(fpath, encoding="utf-8").read()
        else:
            continue
        if raw.strip():
            docs.append({"text": raw, "metadata": {"source": fname}})
    return docs

# ---------- 2. SIMPLE TEXT SPLITTER -------------------------------
def split_text(text: str, size: int, overlap: int) -> List[str]:
    """Character splitter with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

# ---------- 3. EMBEDDING UTILS ------------------------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    """Returns (n, d) float32 numpy array of L2-normalized embeddings."""
    client = openai.OpenAI()
    # OpenAI allows batching up to ~2048 tokens total; we batch by 100 strings
    embs = []
    for i in range(0, len(texts), 100):
        resp = client.embeddings.create(
            input=texts[i : i + 100],
            model=EMB_MODEL)
        embs.extend([d.embedding for d in resp.data])
    arr = np.array(embs, dtype="float32")
    # Normalize for cosine similarity (so we can use inner product index)
    faiss.normalize_L2(arr)
    return arr

# ---------- 4. BUILD & SAVE FAISS INDEX ---------------------------
def create_vector_db(docs: List[Dict]):
    os.makedirs(DB_DIR, exist_ok=True)
    chunks, meta = [], []
    for d in docs:
        for chunk in split_text(d["text"], CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append(chunk)
            meta.append(d["metadata"])
    print(f"→ Embedding {len(chunks)} chunks …")
    embeddings = embed_texts(chunks)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine via inner product
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"texts": chunks, "meta": meta}, f)
    print(f"✅ Vector DB saved to {DB_DIR}")

def load_vector_db():
    if not (os.path.exists(INDEX_FILE) and os.path.exists(META_FILE)):
        sys.exit("No FAISS DB found. Run indexing first.")
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        store = pickle.load(f)
    return index, store["texts"], store["meta"]

# ---------- 5. RETRIEVAL ------------------------------------------
def retrieve(query: str, k: int = 3) -> List[Dict]:
    index, texts, meta = load_vector_db()
    q_emb = embed_texts([query])
    D, I = index.search(q_emb, k)        # distances & indices
    return [{"text": texts[i], "meta": meta[i], "score": float(D[0][rank])}
            for rank, i in enumerate(I[0])]

# ---------- 6. MAIN -----------------------------------------------
if __name__ == "__main__":
    folder = "documents"

    if not (os.path.exists(INDEX_FILE) and os.path.exists(META_FILE)):
        print("No existing index – building one …")
        docs = load_documents(folder)
        if not docs:
            sys.exit(f"No PDFs/TXTs found in '{folder}'.")
        create_vector_db(docs)

    while True:
        q = input("\nQuery (or 'exit'): ")
        if q.lower() == "exit":
            break
        for idx, hit in enumerate(retrieve(q, k=3), 1):
            print(f"\n[{idx}] {hit['meta']['source']}  (score={hit['score']:.3f})\n")
            print(hit["text"])
            print("-" * 60)
