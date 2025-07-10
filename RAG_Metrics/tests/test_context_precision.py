from ragas import evaluate
from ragas.metrics import context_precision
from RAG_Metrics.tests.shared_dataset import dataset

def test_context_precision():
    import numpy as np
    data = dataset.to_list()
    all_scores = []
    failed = []
    threshold = 0.5
    for i, row in enumerate(data):
        single_ds = type(dataset).from_list([row])
        scores = evaluate(single_ds, metrics=[context_precision])
        score = scores["context_precision"][0]
        all_scores.append(score)
        status = "PASS" if score >= threshold else "FAIL"
        from rag_cli import generate_answer
        rag_answer = generate_answer(row["question"])
        reference = row.get("reference", "")
        print(f"Q{i+1}: {row['question']}\n  RAG answer: {rag_answer}\n  Reference: {reference}\n  context_precision: {score:.3f} [{status}]\n")
        if score < threshold:
            failed.append((i+1, row['question'], score, rag_answer, reference))
    avg_score = np.mean(all_scores)
    print(f"Total average context_precision: {avg_score:.3f}\n")
    if failed:
        print("Failed questions:")
        for idx, q, s, rag_answer, reference in failed:
            print(f"  Q{idx}: {q}\n    Score: {s:.3f}\n    RAG answer: {rag_answer}\n    Reference: {reference}")
    else:
        print("All questions passed the threshold.")
    assert avg_score >= threshold