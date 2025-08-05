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

# Self-RAG adaptive stopping config (edit here)
ISSUP_THRESHOLD = 0.8  # Minimum support score (0-1) to stop
ISREL_THRESHOLD = 0.8  # Minimum relevance score (0-1) to stop
ISUSE_THRESHOLD = 0.8  # Minimum usefulness score (0-1) to stop
MAX_ITERATIONS = 3     # Maximum RAG-reflection cycles

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

def generate_answer(query: str) -> str:
    """
    Generate an answer with self-reflection. Adaptive stopping based on ISSUP/ISREL thresholds (0-1).
    Uses global ISSUP_THRESHOLD, ISREL_THRESHOLD, MAX_ITERATIONS.
    Args:
        query (str): The user question.
    Returns:
        str: The final answer.
    """
    import re
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set")
    client = openai.OpenAI()

    def extract_score(section: str) -> float:
        """Extracts a float score from the start of a section string."""
        match = re.match(r"([01](?:\.\d+)?)(?:\s|:|,|$)", section.strip())
        if match:
            try:
                return float(match.group(1))
            except Exception:
                return None
        return None

    def build_context(query: str) -> str:
        """Retrieves and joins top context chunks for the query."""
        hits = retrieve(query)
        return "\n\n".join(h["text"] for h in hits)

    def call_llm(messages, model=LLM_MODEL, temperature=0.2):
        """Calls the LLM with given messages."""
        return client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages
        )

    def parse_critique(critique: str):
        """Extracts and prints each special token section and returns their scores."""
        issup, isrel, isuse = None, None, None
        for token in ["RETRIEVE", "ISSUP", "ISREL", "ISUSE"]:
            match = re.search(rf"{token}:(.*?)(?=\n[A-Z]+:|$)", critique, re.DOTALL)
            if match:
                section = match.group(1).strip()
                print(f"[{token}] {section}")
                if token == "ISSUP":
                    issup = extract_score(section)
                if token == "ISREL":
                    isrel = extract_score(section)
                if token == "ISUSE":
                    isuse = extract_score(section)
        return issup, isrel, isuse

    def print_token_usage(resp):
        if hasattr(resp, "usage") and resp.usage:
            total = getattr(resp.usage, "total_tokens", None)
            prompt = getattr(resp.usage, "prompt_tokens", None)
            completion = getattr(resp.usage, "completion_tokens", None)
            print(f"🪞 Reflection tokens used: total={total}, prompt={prompt}, completion={completion}")
        else:
            print("🪞 Reflection token usage not available.")

    iteration = 0
    answer = None
    while iteration < MAX_ITERATIONS:
        context = build_context(query)
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        # Step 1: Generate answer
        resp = call_llm([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ])
        answer = resp.choices[0].message.content.strip()
        print(f"\n🧠 Iteration {iteration+1} Answer:", answer)

        # Step 2: Critique with special tokens
        critique_prompt = (
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer: {answer}\n\n"
            "Reflect on the answer using the following special tokens as section headers. For each, provide a short explanation.\n"
            "RETRIEVE: What information was retrieved and used?\n"
            "ISSUP: On a scale from 0 to 1, how well is the answer supported by the retrieved context? (Start with a score, e.g., '0.85', then explain)\n"
            "ISREL: On a scale from 0 to 1, how relevant is the answer to the question and context? (Start with a score, e.g., '0.92', then explain)\n"
            "ISUSE: On a scale from 0 to 1, how useful and actionable is the answer for the user? (Start with a score, e.g., '0.90', then explain)\n"
            "Format your response as:\nRETRIEVE: ...\nISSUP: ...\nISREL: ...\nISUSE: ..."
        )
        critique_resp = call_llm([
            {"role": "system", "content": "You are a critical and precise assistant. Only use the provided context."},
            {"role": "user", "content": critique_prompt}
        ])
        critique = critique_resp.choices[0].message.content.strip()
        print("🔍 Critique (with special tokens):\n", critique)

        issup_value, isrel_value, isuse_value = parse_critique(critique)
        print_token_usage(critique_resp)

        # Adaptive stopping: break if ISSUP, ISREL, and ISUSE meet numeric threshold
        if (
            issup_value is not None and issup_value >= ISSUP_THRESHOLD and
            isrel_value is not None and isrel_value >= ISREL_THRESHOLD and
            isuse_value is not None and isuse_value >= ISUSE_THRESHOLD
        ):
            print(f"✅ Stopping: ISSUP={issup_value}, ISREL={isrel_value}, ISUSE={isuse_value} >= threshold (ISSUP={ISSUP_THRESHOLD}, ISREL={ISREL_THRESHOLD}, ISUSE={ISUSE_THRESHOLD}).")
            break
        else:
            print(f"🔄 Continuing: ISSUP={issup_value}, ISREL={isrel_value}, ISUSE={isuse_value} (thresholds: ISSUP={ISSUP_THRESHOLD}, ISREL={ISREL_THRESHOLD}, ISUSE={ISUSE_THRESHOLD}).")
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
