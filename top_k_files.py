

import os, pickle
from typing import List, Tuple

import fitz                   # PyMuPDF  →  pip install pymupdf
import numpy as np
import faiss                  # pip install faiss-cpu
import openai                 # pip install openai

DOCS_DIR        = "documents"
DOC_INDEX_FILE  = "doc_index.faiss"
DOC_META_FILE   = "doc_meta.pkl"
EMBED_MODEL     = "text-embedding-3-small"   # adjust if needed


# ── helpers ──────────────────────────────────────────────────────────────
def extract_text_from_pdf(path: str) -> str:
    text = []
    with fitz.open(path) as doc:
        for page in doc:
            text.append(page.get_text("text"))
    return "\n".join(text)


def embed_texts(texts: List[str]) -> np.ndarray:
    client = openai.OpenAI()                   # requires OPENAI_API_KEY env-var
    out = []
    for i in range(0, len(texts), 100):        # batch ≤ 100 inputs
        resp = client.embeddings.create(input=texts[i:i+100], model=EMBED_MODEL)
        out.extend([d.embedding for d in resp.data])
    vecs = np.array(out, dtype="float32")
    faiss.normalize_L2(vecs)                   # cosine → inner product
    return vecs


# ── index build / load ───────────────────────────────────────────────────
def build_doc_index(folder: str = DOCS_DIR) -> None:
    texts, names = [], []
    for fn in os.listdir(folder):
        fp = os.path.join(folder, fn)
        if fn.lower().endswith(".pdf"):
            texts.append(extract_text_from_pdf(fp))
            names.append(fn)
        elif fn.lower().endswith(".txt"):
            texts.append(open(fp, encoding="utf-8").read())
            names.append(fn)

    if not texts:
        raise ValueError("No PDFs/TXTs found to index.")

    print(f"Embedding {len(texts)} documents …")
    vecs = embed_texts(texts)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    faiss.write_index(index, DOC_INDEX_FILE)
    with open(DOC_META_FILE, "wb") as f:
        pickle.dump(names, f)
    print("✅ doc-level FAISS index saved.")


def load_doc_index():
    if not os.path.exists(DOC_INDEX_FILE):
        build_doc_index()                # auto-build on first use
    index = faiss.read_index(DOC_INDEX_FILE)
    names = pickle.load(open(DOC_META_FILE, "rb"))
    return index, names


# ── public API ───────────────────────────────────────────────────────────
def top_k_docs(query: str, k: int = 5) -> List[Tuple[str, float]]:
    index, names = load_doc_index()
    qvec = embed_texts([query])
    D, I = index.search(qvec, k)
    return [(names[i], float(D[0][rank])) for rank, i in enumerate(I[0])]


# ── demo ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for fname, score in top_k_docs("wireless charging on Galaxy S22", k=3):
        print(f"{fname}  (score={score:.3f})")
