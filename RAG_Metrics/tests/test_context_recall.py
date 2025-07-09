from ragas import evaluate
from ragas.metrics import context_recall
from .shared_dataset import dataset

def test_context_recall():
    import numpy as np
    data = dataset.to_list()
    all_scores = []
    failed = []
    threshold = 0.5
    for i, row in enumerate(data):
        single_ds = type(dataset).from_list([row])
        scores = evaluate(single_ds, metrics=[context_recall])
        score = scores["context_recall"][0]
        all_scores.append(score)
        status = "PASS" if score >= threshold else "FAIL"
        reference = row.get("reference", "")
        contexts = row.get("contexts", [])
        print(f"Q{i+1}: {row['question']}\n  Reference: {reference}\n  Retrieved Contexts: {contexts}\n  context_recall: {score:.3f} [{status}]\n")
        if score < threshold:
            failed.append((i+1, row['question'], score, reference, contexts))
    avg_score = np.mean(all_scores)
    print(f"Total average context_recall: {avg_score:.3f}\n")
    if failed:
        print("Failed questions:")
        for idx, q, s, reference, contexts in failed:
            print(f"  Q{idx}: {q}\n    Score: {s:.3f}\n    Reference: {reference}\n    Retrieved Contexts: {contexts}")
    else:
        print("All questions passed the threshold.")
    assert avg_score >= threshold
