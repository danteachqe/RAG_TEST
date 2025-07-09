from ragas import evaluate
from ragas.metrics import faithfulness
from RAG_Metrics.tests.shared_dataset import dataset
from RAG_Metrics.rag_cli import generate_answer


def test_truthfulness():
    import numpy as np
    from datasets import Dataset

    data = dataset.to_list()
    all_scores = []
    failed = []
    threshold = 0.5

    for i, row in enumerate(data):
        # Step 1: Generate answer
        rag_answer = generate_answer(row["question"])

        # Step 2: Build enriched input for Ragas
        enriched_row = {
            "question": row["question"],
            "contexts": row["contexts"],
            "answer": rag_answer,
            "reference": row["reference"]
        }

        # Step 3: Evaluate with faithfulness metric
        single_ds = Dataset.from_list([enriched_row])
        scores = evaluate(single_ds, metrics=[faithfulness])
        score = scores["faithfulness"][0]
        all_scores.append(score)
        status = "PASS" if score >= threshold else "FAIL"

        # Step 4: Display results
        print(f"\nQ{i+1}: {row['question']}")
        print(f"  RAG answer     : {rag_answer}")
        print(f"  Reference      : {row['reference']}")
        print(f"  truthfulness   : {score:.3f} [{status}]")

        print("  Contexts:")
        for j, ctx in enumerate(enriched_row["contexts"], 1):
            snippet = ctx.strip().replace("\n", " ")
            print(f"    [{j}] {snippet[:200]}{'...' if len(snippet) > 200 else ''}")

        # Step 5: Track failures
        if score < threshold:
            failed.append((i+1, row["question"], score, rag_answer, row["reference"]))

    # Step 6: Compute average safely
    avg_score = np.nanmean(all_scores)
    print(f"\nTotal average truthfulness: {avg_score:.3f}")

    # Step 7: Summary
    if failed:
        print("\n❌ Failed questions:")
        for idx, q, s, rag_answer, reference in failed:
            print(f"  Q{idx}: {q}\n    Score: {s:.3f}\n    RAG answer: {rag_answer}\n    Reference: {reference}")
    else:
        print("✅ All questions passed the threshold.")

    # Step 8: Assertion
    assert not np.isnan(avg_score), "Average score is NaN — likely missing answer or context."
    assert avg_score >= threshold, f"Average truthfulness {avg_score:.3f} is below threshold {threshold}"
