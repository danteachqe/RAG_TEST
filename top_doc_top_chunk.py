def two_stage_retrieve(query: str) -> List[Dict]:
    """
    Step 1: Retrieve top-1 most relevant document.
    Step 2: Chunk that document and retrieve top-3 chunks.
    Returns: list of top 3 chunks with score and metadata.
    """
    # Step 1 – retrieve top-1 document
    top_docs = top_k_docs(query, k=1)  # [(filename, score)]
    doc_path = os.path.join(DOCS_DIR, top_docs[0][0])

    # Step 2 – load and split the document
    if doc_path.endswith(".pdf"):
        full_text = extract_text_from_pdf(doc_path)
    elif doc_path.endswith(".txt"):
        full_text = open(doc_path, encoding="utf-8").read()
    else:
        return []

    chunks = split_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
    embeddings = embed_texts(chunks)

    # Step 3 – search top-3 chunks
    faiss.normalize_L2(embeddings)
    query_vec = embed_texts([query])
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    D, I = index.search(query_vec, 3)

    return [{
        "chunk": chunks[i],
        "doc": top_docs[0][0],
        "doc_score": top_docs[0][1],
        "chunk_score": float(D[0][rank])
    } for rank, i in enumerate(I[0])]
