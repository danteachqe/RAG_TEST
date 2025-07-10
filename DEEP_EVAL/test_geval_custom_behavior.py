# test_geval_s22_quality.py

from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from simple_rag import ask_llm

def build_metrics():
    return [
        GEval(
            name="Fluency",
            criteria="Is the output grammatically correct and easy to understand?",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
        ),
        GEval(
            name="Coherence",
            criteria="Is the output logically structured and cohesive?",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.5
        ),
        GEval(
            name="Relevance",
            criteria="Does the output appropriately and directly answer the input?",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.5
        ),
        GEval(
            name="Concision",
            criteria="Is the output concise, avoiding redundancy and unnecessary verbosity while preserving meaning?",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.5
        ),
        GEval(
            name="Technical Accuracy",
            criteria="Is the answer technically accurate based on known Galaxy S22 Ultra specifications?",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.5
        ),
    ]

def test_display_resolutions():
    query = "What display resolutions can I set on the Galaxy S22 Ultra?"
    actual_output = ask_llm(query)
    test_case = LLMTestCase(input=query, actual_output=actual_output)
    assert_test(test_case, build_metrics())

def test_camera_features():
    query = "What camera capabilities does the Galaxy S22 Ultra offer?"
    actual_output = ask_llm(query)
    test_case = LLMTestCase(input=query, actual_output=actual_output)
    assert_test(test_case, build_metrics())

def test_s_pen_functionality():
    query = "What can I do with the S Pen on the Galaxy S22 Ultra?"
    actual_output = ask_llm(query)
    test_case = LLMTestCase(input=query, actual_output=actual_output)
    assert_test(test_case, build_metrics())
