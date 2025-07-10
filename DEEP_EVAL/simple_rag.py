from __future__ import annotations

import os, pickle, sys
from typing import List, Dict

import numpy as np
import fitz                      # PyMuPDF
import faiss                     # Facebook AI Similarity Search
from tqdm import tqdm
import openai                    # for embeddings *and* chat

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DOCS_DIR     = "documents"        # Folder with source docs
DB_DIR       = "faiss_index"      # Where the index lives
INDEX_FILE   = os.path.join(DB_DIR, "index.faiss")
META_FILE    = os.path.join(DB_DIR, "docs.pkl")

CHUNK_SIZE    = 1500
CHUNK_OVERLAP = 300
EMBED_MODEL   = "text-embedding-3-small"
MAX_CONTEXTS  = 3

# Chat‑completion model for generation
LLM_MODEL     = "gpt-4o-mini"     # change to any available model id

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
# 5. Build prompt, generate answer
# ─────────────────────────────────────────────────────────────────────────────

def ask_llm(query: str):
    hits = retrieve(query)

    # 1) Print contexts
    print("\n🔍 Retrieved Contexts:\n")
    for rank, h in enumerate(hits, 1):
        src, score = h["meta"].get("source", "<unknown>"), h["score"]
        print(f"[{rank}] {src} (score={score:.3f})\n{h['text']}\n")

    # 2) Assemble prompt
    context_block = "\n\n".join(h["text"] for h in hits)
    user_prompt = (
        "Context:\n" + context_block + "\n\n" +
        f"Question: {query}\nAnswer:"
    )

    print("📝 Prompt sent to LLM:\n")
    full_prompt_preview = SYSTEM_PROMPT + "\n\n" + user_prompt
    print(full_prompt_preview)

    # 3) Generate answer
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set.")

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )
    answer = resp.choices[0].message.content.strip()

    print("\n💬 Assistant Answer:\n")
    print(answer)
    return answer 

# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Complete RAG CLI — type 'exit' to quit\n")
    while True:
        q = input(">>> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        try:
            ask_llm(q)
        except Exception as exc:
            print("Error:", exc, "\n")

if __name__ == "__main__":
    main()
