from ragas import evaluate
from ragas.metrics import answer_relevancy
from RAG_Metrics.tests.shared_dataset import dataset

def test_answer_relevancy():
    import numpy as np
    from RAG_Metrics.rag_cli import generate_answer
    from datasets import Dataset
    from openai import OpenAI

    openai_client = OpenAI()
    data = dataset.to_list()
    all_scores = []
    failed = []
    threshold = 0.5

    details = []
    for i, row in enumerate(data):
        rag_answer = generate_answer(row["question"])
        # Inject required fields for ragas
        enriched_row = {
            "question": row["question"],
            "answer": rag_answer,
            "contexts": row["contexts"],
            "ground_truths": [row["reference"]],  # ✅ injected dynamically
            "reference": row["reference"]         # still used for display
        }

        single_ds = Dataset.from_list([enriched_row])
        scores = evaluate(single_ds, metrics=[answer_relevancy])
        score = scores["answer_relevancy"][0]
        all_scores.append(score)

        # LLM reasoning
        system_prompt = "You are a helpful evaluator judging relevance."
        sub_questions = [
            "Does the answer attempt to address the user's question?",
            "Does the answer provide relevant and helpful information in response to the question?",
            "Is the answer accurate and correct in the context of the question?"
        ]
        analysis = ""
        for sq in sub_questions:
            full_prompt = f"{sq}\n\nQuestion: {row['question']}\nAnswer: {rag_answer}"
            chat = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ]
            )
            analysis += f"\n- {sq}\n  → {chat.choices[0].message.content.strip()}\n"

        status = "PASS" if score >= threshold else "FAIL"
        details.append({
            "idx": i+1,
            "question": row["question"],
            "rag_answer": rag_answer,
            "reference": row["reference"],
            "score": score,
            "status": status,
            "analysis": analysis
        })

        if score < threshold:
            failed.append((i+1, row['question'], score, rag_answer, row["reference"]))

    for d in details:
        print(f"\nQ{d['idx']}: {d['question']}")
        print(f"  RAG answer      : {d['rag_answer']}")
        print(f"  Ground truth    : {d['reference']}")
        print(f"  answer_relevancy: {d['score']:.3f} [{d['status']}]")
        print(f"  Reasoning       :{d['analysis']}")

    avg_score = np.mean(all_scores)
    print(f"\nTotal average answer_relevancy: {avg_score:.3f}")

    if failed:
        print("\n❌ Failed questions:")
        for idx, q, s, rag_answer, reference in failed:
            print(f"  Q{idx}: {q}\n    Score: {s:.3f}\n    RAG answer: {rag_answer}\n    Reference: {reference}")
    else:
        print("✅ All questions passed the threshold.")

    assert avg_score >= threshold
