# Simple Retriever using FAISS and OpenAI Embeddings + Eval Harness
import os, pickle, sys, json, time, math, re
from typing import List, Dict, Any, Tuple
import numpy as np
import fitz                         # PyMuPDF
import faiss                       # Facebook AI Similarity Search
from tqdm import tqdm
import statistics
import argparse

import openai

# -------------------------- Config ---------------------------------
# One level above current script folder
DB_DIR        = os.path.join("..", "faiss_index")
DOCS_DIR      = os.path.join("..", "Documents")

INDEX_FILE    = os.path.join(DB_DIR, "index.faiss")
META_FILE     = os.path.join(DB_DIR, "docs.pkl")

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
EMB_MODEL     = "text-embedding-3-small"
# -------------------------------------------------------------------


# Eval defaults
DEFAULT_K                 = 5
LEAKAGE_SCORE_THRESHOLD   = 0.20    # inner-product (cosine) score above which we consider "confident"
REDUNDANCY_SIM_THRESHOLD  = 0.90    # Jaccard token similarity to treat two chunks as near-duplicates
NDCG_K                    = 5
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
        start += max(1, size - overlap)
    return chunks

# ---------- 3. EMBEDDING UTILS ------------------------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    """Returns (n, d) float32 numpy array of L2-normalized embeddings."""
    client = openai.OpenAI()
    embs = []
    for i in range(0, len(texts), 100):
        resp = client.embeddings.create(
            input=texts[i : i + 100],
            model=EMB_MODEL
        )
        embs.extend([d.embedding for d in resp.data])
    arr = np.array(embs, dtype="float32")
    faiss.normalize_L2(arr)  # for cosine via inner product index
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
    hits = []
    for rank, i in enumerate(I[0]):
        if i == -1:
            continue
        hits.append({
            "text": texts[i],
            "meta": meta[i],
            "score": float(D[0][rank]),
            "idx": int(i)
        })
    return hits

# ---------- 6. METRICS --------------------------------------------

def token_set(s: str) -> set:
    # simple tokenization, lowercase, alnum only
    toks = re.findall(r"[0-9A-Za-zÀ-ž]+", s.lower())
    # optional: cheap stopwords
    stop = {"the","a","an","and","or","de","la","si","și","care","cu","din","in","în","pe","pentru"}
    return {t for t in toks if t not in stop}

def jaccard(a: str, b: str) -> float:
    A, B = token_set(a), token_set(b)
    if not A and not B: return 1.0
    if not A or not B:  return 0.0
    return len(A & B) / len(A | B)

def is_relevant(hit: Dict, gold: Dict) -> bool:
    """
    A hit is considered relevant if:
      - its source is in gold_sources, OR
      - its text contains any of gold_substrings (case-insensitive)
    """
    src_ok = False
    sub_ok = False
    if "gold_sources" in gold and gold["gold_sources"]:
        src = hit["meta"].get("source", "")
        src_ok = src in set(gold["gold_sources"])
    if "gold_substrings" in gold and gold["gold_substrings"]:
        ht = hit["text"].lower()
        sub_ok = any(sub.lower() in ht for sub in gold["gold_substrings"])
    return src_ok or sub_ok

def precision_at_k(hits: List[Dict], gold: Dict) -> float:
    if not hits: return 0.0
    rel = sum(1 for h in hits if is_relevant(h, gold))
    return rel / len(hits)

def recall_at_k(hits: List[Dict], gold: Dict) -> float:
    # Binary recall: did we retrieve at least one relevant?
    return 1.0 if any(is_relevant(h, gold) for h in hits) else 0.0

def reciprocal_rank(hits: List[Dict], gold: Dict) -> float:
    for i, h in enumerate(hits, 1):
        if is_relevant(h, gold):
            return 1.0 / i
    return 0.0

def ndcg_at_k(hits: List[Dict], gold: Dict, k: int = 5) -> float:
    # Binary relevance unless graded info is provided via gold_substrings_graded (dict substring->grade)
    grades = {}
    if gold.get("gold_substrings_graded"):
        for sub, g in gold["gold_substrings_graded"].items():
            grades[sub.lower()] = float(g)
    def gain(h: Dict) -> float:
        if grades:
            ht = h["text"].lower()
            gmax = 0.0
            for sub, g in grades.items():
                if sub in ht:
                    gmax = max(gmax, g)
            # source-based graded gain: optional
            if "gold_sources_graded" in gold:
                src = h["meta"].get("source","")
                gmax = max(gmax, float(gold["gold_sources_graded"].get(src, 0.0)))
            return gmax
        # binary
        return 1.0 if is_relevant(h, gold) else 0.0
    # DCG
    dcg = 0.0
    for i, h in enumerate(hits[:k], start=1):
        dcg += (2**gain(h) - 1) / math.log2(i + 1)
    # IDCG (ideal)
    # make a multiset of potential relevant items: we approximate by sorting gains of retrieved items
    # If you have full graded labels per collection, replace with true IDCG.
    gains_sorted = sorted([(2**gain(h)-1) for h in hits[:k]], reverse=True)
    idcg = 0.0
    for i, g in enumerate(gains_sorted, start=1):
        idcg += g / math.log2(i + 1)
    return (dcg / idcg) if idcg > 0 else 0.0

def unique_ratio_topk(hits: List[Dict], thr: float = REDUNDANCY_SIM_THRESHOLD) -> float:
    if not hits: return 1.0
    kept = []
    for h in hits:
        t = h["text"]
        if all(jaccard(t, other["text"]) < thr for other in kept):
            kept.append(h)
    return len(kept) / len(hits)

# ---------- 7. EVAL HARNESS ---------------------------------------

def eval_file_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

def timed_retrieve(query: str, k: int) -> Tuple[List[Dict], float]:
    t0 = time.perf_counter()
    hits = retrieve(query, k=k)
    t1 = time.perf_counter()
    return hits, (t1 - t0)

def evaluate(eval_path: str, k: int = DEFAULT_K) -> None:
    rows = eval_file_rows(eval_path)

    # Buckets
    queries = [r for r in rows if not r.get("nonsense")]
    nonsense_queries = [r for r in rows if r.get("nonsense")]

    # Metrics collectors
    precs, recs, mrrs, ndcgs, uniqs, latencies = [], [], [], [], [], []
    # For coverage
    key_docs = set()
    retrieved_relevant_docs = set()

    # Paraphrase / language robustness
    para_deltas = []     # (recall_base - recall_para)
    lang_deltas = []     # (recall_base - recall_var)

    # Main loop: real queries
    for r in tqdm(queries, desc="Evaluating queries"):
        q = r["query"]
        hits, dt = timed_retrieve(q, k=k)
        latencies.append(dt)

        # Core metrics
        precs.append(precision_at_k(hits, r))
        recs.append(recall_at_k(hits, r))
        mrrs.append(reciprocal_rank(hits, r))
        ndcgs.append(ndcg_at_k(hits, r, k=NDCG_K))
        uniqs.append(unique_ratio_topk(hits))

        # Coverage bookkeeping
        if "gold_sources" in r:
            for s in r["gold_sources"]:
                key_docs.add(s)
        # mark which relevant docs we actually retrieved
        for h in hits:
            if is_relevant(h, r):
                src = h["meta"].get("source","")
                if src:
                    retrieved_relevant_docs.add(src)

        # Paraphrase robustness (optional)
        if r.get("paraphrases"):
            base = recall_at_k(hits, r)
            for p in r["paraphrases"]:
                phits, _ = timed_retrieve(p, k=k)
                para = recall_at_k(phits, r)
                para_deltas.append(base - para)

        # Language/diacritics robustness (optional)
        if r.get("variants"):
            base = recall_at_k(hits, r)
            for v in r["variants"]:
                vhits, _ = timed_retrieve(v, k=k)
                var = recall_at_k(vhits, r)
                lang_deltas.append(base - var)

    # Out-of-scope leakage
    leakage_count = 0
    total_nonsense = len(nonsense_queries)
    for r in tqdm(nonsense_queries, desc="Evaluating out-of-scope"):
        hits, _ = timed_retrieve(r["query"], k=k)
        # "confident" hit = any score above threshold
        if any(h["score"] >= LEAKAGE_SCORE_THRESHOLD for h in hits):
            leakage_count += 1
    leakage_rate = (leakage_count / total_nonsense) if total_nonsense > 0 else 0.0

    # Aggregations
    precision = sum(precs)/len(precs) if precs else 0.0
    recall    = sum(recs)/len(recs) if recs else 0.0
    mrr       = sum(mrrs)/len(mrrs) if mrrs else 0.0
    ndcg      = sum(ndcgs)/len(ndcgs) if ndcgs else 0.0
    uniq      = sum(uniqs)/len(uniqs) if uniqs else 1.0

    # Coverage
    coverage = (len(retrieved_relevant_docs & key_docs) / len(key_docs)) if key_docs else 0.0

    # Latency p95
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0

    # Robustness deltas
    para_drop = (sum(para_deltas)/len(para_deltas)) if para_deltas else 0.0
    lang_drop = (sum(lang_deltas)/len(lang_deltas)) if lang_deltas else 0.0

    # --------- Summary printout ----------
    print("\n" + "="*72)
    print("RAG RETRIEVAL — 10 METRICS SUMMARY")
    print("="*72)
    print(f"k                                    : {k}")
    print(f"Queries (non-nonsense)               : {len(queries)}")
    print(f"Nonsense queries                     : {total_nonsense}")
    print("-"*72)
    print(f"1) Recall@k                          : {recall:.3f}")
    print(f"2) Precision@k                       : {precision:.3f}")
    print(f"3) MRR                               : {mrr:.3f}")
    print(f"4) Coverage                          : {coverage*100:.1f}%")
    print(f"5) nDCG@{NDCG_K}                         : {ndcg:.3f}")
    print(f"6) Latency p95 (s)                   : {p95:.3f}")
    print(f"7) Paraphrase robustness (Δ Recall)  : {para_drop:+.3f}  (<= +0.050 recommended)")
    print(f"8) Language/diacritics (Δ Recall)    : {lang_drop:+.3f}  (<= +0.050 recommended)")
    print(f"9) Redundancy (unique ratio top-k)   : {uniq:.3f}       (>= 0.80 recommended)")
    print(f"10) Out-of-scope leakage rate        : {leakage_rate*100:.2f}% (<= 1.00% recommended)")
    print("="*72 + "\n")

# ---------- 8. MAIN -----------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=str, default=None, help="Path to eval.jsonl")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-k to retrieve")
    parser.add_argument("--folder", type=str, default=DOCS_DIR, help="Docs folder")
    args = parser.parse_args()

    # Build index if missing
    if not (os.path.exists(INDEX_FILE) and os.path.exists(META_FILE)):
        print("No existing index – building one …")
        docs = load_documents(args.folder)
        if not docs:
            sys.exit(f"No PDFs/TXTs found in '{args.folder}'.")
        create_vector_db(docs)

    if args.eval:
        evaluate(args.eval, k=args.k)
        return

    # Interactive mode
    while True:
        q = input("\nQuery (or 'exit'): ")
        if q.lower() == "exit":
            break
        hits = retrieve(q, k=args.k)
        for idx, hit in enumerate(hits, 1):
            print(f"\n[{idx}] {hit['meta'].get('source','?')}  (score={hit['score']:.3f})\n")
            print(hit["text"])
            print("-" * 60)

if __name__ == "__main__":
    main()

# python .\retrieval_full_test.py --eval ".\eval.jsonl" --k 5