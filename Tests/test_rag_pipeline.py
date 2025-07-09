import pytest
from rag_pipeline import generate_answer

@pytest.mark.parametrize("query, expected_keywords", [
    ("What is S22?", ["phone", "samsung"]),
    ("Does Galaxy S22 support wireless charging?", ["wireless", "charging"]),
])
def test_rag_pipeline_generation(query, expected_keywords):
    answer = generate_answer(query)
    assert answer, "No answer generated."
    for keyword in expected_keywords:
        assert keyword.lower() in answer.lower(), f"Missing keyword: {keyword}"
