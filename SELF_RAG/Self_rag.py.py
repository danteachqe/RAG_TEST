import os, pickle, sys
from typing import List, Dict
import numpy as np
import fitz  # PyMuPDF
import faiss
from tqdm import tqdm
import openai

# ─────────────────────────────────────
# CONFIG
# ─────────────────────────────────────
DOCS_DIR = r"C:\code\RAG_Test\RAG_TEST\Documents"

DB_DIR       = "faiss_index"
INDEX_FILE   = os.path.join(DB_DIR, "index.faiss")
META_FILE    = os.path.join(DB_DIR, "docs.pkl")

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
EMBED_MODEL   = "text-embedding-3-small"
MAX_CONTEXTS  = 3
LLM_MODEL     = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a concise, highly accurate assistant. "
    "If the answer cannot be found in the provided context, say 'I don't know.'"
)

# ─────────────────────────────────────
# DOCUMENT LOADING
# ─────────────────────────────────────
def extract_text_from_pdf(path: str) -> str:
    out = []
    with fitz.open(path) as doc:
        for page in doc:
            out.append(page.get_text("text"))
    return "\n".join(out)

def load_documents(folder: str = DOCS_DIR) -> List[Dict]:
    docs = []
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

# ─────────────────────────────────────
# CHUNKING & EMBEDDING
# ─────────────────────────────────────
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
    all_vecs = []
    batch = 100
    for i in tqdm(range(0, len(texts), batch), desc="Embedding", leave=False):
        resp = client.embeddings.create(input=texts[i:i+batch], model=model)
        all_vecs.extend([d.embedding for d in resp.data])
    arr = np.array(all_vecs, dtype="float32")
    faiss.normalize_L2(arr)
    return arr

# ─────────────────────────────────────
# VECTOR DB
# ─────────────────────────────────────
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

# ─────────────────────────────────────
# RETRIEVE + GENERATE
# ─────────────────────────────────────
def retrieve(query: str, k: int = MAX_CONTEXTS):
    index, texts, meta = load_vector_db()
    q_vec = embed_texts([query])
    D, I = index.search(q_vec, k)
    return [{"text": texts[i], "meta": meta[i], "score": float(D[0][rank])}
            for rank, i in enumerate(I[0])]

def generate_answer(query: str, issup_threshold=0.8, isrel_threshold=0.8, max_iterations=3) -> str:
    """
    Generate an answer with self-reflection. Adaptive stopping based on ISSUP/ISREL thresholds (0-1).
    Args:
        query (str): The user question.
        issup_threshold (float): Minimum support score (0-1) to stop.
        isrel_threshold (float): Minimum relevance score (0-1) to stop.
        max_iterations (int): Maximum RAG-reflection cycles.
    Returns:
        str: The final answer.
    """
    import re
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set")
    client = openai.OpenAI()

    def extract_score(section):
        match = re.match(r"([01](?:\.\d+)?)(?:\s|:|,|$)", section.strip())
        if match:
            try:
                return float(match.group(1))
            except Exception:
                return None
        return None

    iteration = 0
    answer = None
    while iteration < max_iterations:
        hits = retrieve(query)
        context = "\n\n".join(h["text"] for h in hits)
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        # Step 1: Generate answer
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        )
        answer = resp.choices[0].message.content.strip()
        print(f"\n🧠 Iteration {iteration+1} Answer:", answer)

        # Step 2: Critique with special tokens
        critique_prompt = (
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer: {answer}\n\n"
            "Reflect on the answer using the following special tokens as section headers. For each, provide a short explanation.\n"
            "RETRIEVE: What information was retrieved and used?\n"
            "ISSUP: Is the answer supported by the retrieved context? (Answer 'yes' or 'no' at the start, then explain)\n"
            "ISREL: Is the answer relevant to the question and context? (Answer 'yes' or 'no' at the start, then explain)\n"
            "ISUSE: Is the answer useful and actionable for the user?\n"
            "Format your response as:\nRETRIEVE: ...\nISSUP: ...\nISREL: ...\nISUSE: ..."
        )
        critique_resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a critical and precise assistant. Only use the provided context."},
                {"role": "user", "content": critique_prompt}
            ]
        )
        critique = critique_resp.choices[0].message.content.strip()
        print("🔍 Critique (with special tokens):\n", critique)

        # Parse and print each section separately
        issup_value, isrel_value = None, None
        for token in ["RETRIEVE", "ISSUP", "ISREL", "ISUSE"]:
            match = re.search(rf"{token}:(.*?)(?=\n[A-Z]+:|$)", critique, re.DOTALL)
            if match:
                section = match.group(1).strip()
                print(f"[{token}] {section}")
                if token == "ISSUP":
                    issup_value = section.split(" ")[0].strip().lower()
                if token == "ISREL":
                    isrel_value = section.split(" ")[0].strip().lower()

        # Print token usage for the reflection step if available
        if hasattr(critique_resp, "usage") and critique_resp.usage:
            total_tokens = getattr(critique_resp.usage, "total_tokens", None)
            prompt_tokens = getattr(critique_resp.usage, "prompt_tokens", None)
            completion_tokens = getattr(critique_resp.usage, "completion_tokens", None)
            print(f"🪞 Reflection tokens used: total={total_tokens}, prompt={prompt_tokens}, completion={completion_tokens}")
        else:
            print("🪞 Reflection token usage not available.")

        # Adaptive stopping: break if ISSUP and ISREL meet threshold
        if issup_value == issup_threshold and isrel_value == isrel_threshold:
            print(f"✅ Stopping: ISSUP and ISREL both '{issup_threshold}'.")
            break
        else:
            print(f"🔄 Continuing: ISSUP='{issup_value}', ISREL='{isrel_value}' (threshold='{issup_threshold}').")
        iteration += 1
    return answer

# ─────────────────────────────────────
# OPTIONAL: setup index if missing
# ─────────────────────────────────────
def setup():
    if not os.path.exists(INDEX_FILE):
        docs = load_documents()
        if not docs:
            sys.exit(f"No PDFs/TXTs found in '{DOCS_DIR}'.")
        create_vector_db(docs)

# ─────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────
if __name__ == "__main__":
    setup()
    print("✅ RAG Ready. Ask your question (type 'exit' to quit):")
    while True:
        q = input(">>> ").strip()
        if q.lower() == "exit":
            print("👋 Exiting.")
            break
        try:
            a = generate_answer(q)
            print("🧠 Answer:", a)
        except Exception as e:
            print("❌ Error:", e)
