"""
rag_cli.py — Retrieval & Prompt Builder (no generation)
======================================================
• Loads PDFs/TXTs from *documents*.
• Builds/loads a FAISS index with OpenAI embeddings.
• CLI: retrieves top-k chunks for a question, prints them **and** prints
  the prompt that would be sent to an LLM — but **stops there** (no
  chat-completion call).

Run
---
```
python rag_cli.py
```
Type a question or `exit` to quit.
"""

from __future__ import annotations

import os, pickle, sys
from typing import List, Dict

import numpy as np
import fitz                      # PyMuPDF
import faiss                     # Facebook AI Similarity Search
from tqdm import tqdm
import openai                    # still used for embeddings, not for chat

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DOCS_DIR     = "documents"        # Folder with source docs
DB_DIR       = "faiss_index"      # Where the index lives
INDEX_FILE   = os.path.join(DB_DIR, "index.faiss")
META_FILE    = os.path.join(DB_DIR, "docs.pkl")

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
EMBED_MODEL   = "text-embedding-3-small"
MAX_CONTEXTS  = 3                  # chunks to display
SYSTEM_PROMPT = (
    "You are a concise, highly accurate assistant. "
    "If the answer cannot be found in the provided context, say 'I don't know.'"
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Document loading
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> str:
    out = []
    with fitz.open(path) as doc:
        for page in doc:
            out.append(page.get_text("text"))
    return "\n".join(out)

def load_documents(folder: str = DOCS_DIR) -> List[Dict]:
    docs: List[Dict] = []
    for fn in os.listdir(folder):
        fp = os.path.join(folder, fn)
        if fn.lower().endswith(".pdf"):
            raw = extract_text_from_pdf(fp)
        elif fn.lower().endswith(".txt"):
            with open(fp, encoding="utf-8") as f:
                raw = f.read()
        else:
            continue
        if raw.strip():
            docs.append({"text": raw, "metadata": {"source": fn}})
    return docs

# ─────────────────────────────────────────────────────────────────────────────
# 2. Chunking & embeddings
# ─────────────────────────────────────────────────────────────────────────────

def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def embed_texts(texts: List[str], model: str = EMBED_MODEL) -> np.ndarray:
    client = openai.OpenAI()
    all_vecs: List[List[float]] = []
    batch = 100
    for i in tqdm(range(0, len(texts), batch), desc="Embedding", leave=False):
        resp = client.embeddings.create(input=texts[i:i+batch], model=model)
        all_vecs.extend([d.embedding for d in resp.data])
    arr = np.array(all_vecs, dtype="float32")
    faiss.normalize_L2(arr)
    return arr

# ─────────────────────────────────────────────────────────────────────────────
# 3. Build / load FAISS
# ─────────────────────────────────────────────────────────────────────────────

def create_vector_db(docs: List[Dict]):
    os.makedirs(DB_DIR, exist_ok=True)
    chunks, metas = [], []
    for d in docs:
        for ch in split_text(d["text"]):
            chunks.append(ch)
            metas.append(d["metadata"])
    print(f"Embedding {len(chunks)} chunks …")
    vecs = embed_texts(chunks)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"texts": chunks, "meta": metas}, f)
    print("✅ Vector DB built at", DB_DIR)

def load_vector_db():
    if not (os.path.exists(INDEX_FILE) and os.path.exists(META_FILE)):
        raise FileNotFoundError("FAISS DB not found.")
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        store = pickle.load(f)
    return index, store["texts"], store["meta"]

# Build if missing
if not os.path.exists(INDEX_FILE):
    print("No index found; building one now …")
    docs = load_documents()
    if not docs:
        sys.exit(f"No PDFs/TXTs found in '{DOCS_DIR}'.")
    create_vector_db(docs)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(query: str, k: int = MAX_CONTEXTS):
    index, texts, meta = load_vector_db()
    q_vec = embed_texts([query])
    D, I = index.search(q_vec, k)
    return [{"text": texts[i], "meta": meta[i], "score": float(D[0][rank])}
            for rank, i in enumerate(I[0])]

# ─────────────────────────────────────────────────────────────────────────────
# 5. Build and show prompt (no generation)
# ─────────────────────────────────────────────────────────────────────────────

def show_prompt(query: str):
    hits = retrieve(query)

    print("\n🔍 Retrieved Contexts:\n")
    for rank, h in enumerate(hits, 1):
        source = h["meta"].get("source", "<unknown>")
        score  = h["score"]
        print(f"[{rank}] {source} (score={score:.3f})\n{h['text']}\n")

    context_block = "\n\n".join(h["text"] for h in hits)
    prompt = (
        SYSTEM_PROMPT + "\n\n" +
        "Context:\n" + context_block + "\n\n" +
        f"Question: {query}\nAnswer:"
    )
    print("📝 Prompt that would be sent to the LLM:\n")
    print(prompt)

# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("RAG prompt-builder CLI — type 'exit' to quit\n")
    while True:
        q = input(">>> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        try:
            show_prompt(q)
        except Exception as exc:
            print("Error:", exc, "\n")

if __name__ == "__main__":
    main()
