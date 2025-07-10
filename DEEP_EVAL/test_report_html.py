# test_geval_s22_html.py

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from simple_rag import ask_llm

import pandas as pd
import os

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
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
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

def log_results_to_html(test_name, query, actual_output):
    test_case = LLMTestCase(input=query, actual_output=actual_output)
    data = []

    for metric in build_metrics():
        score = metric.measure(test_case)
        data.append({
            "Test Name": test_name,
            "Query": query,
            "Metric": metric.name,
            "Score": round(score, 3),
            "Passed": score >= metric.threshold if metric.threshold else "N/A",
            "Explanation": ""  # No explanation available in older versions
        })


    df = pd.DataFrame(data)
    report_file = "deepeval_report.html"

    if not os.path.exists(report_file):
        df.to_html(report_file, index=False)
    else:
        with open(report_file, "a", encoding="utf-8") as f:
            f.write(df.to_html(index=False, header=False))

def test_display_resolutions():
    query = "What display resolutions can I set on the Galaxy S22 Ultra?"
    actual_output = ask_llm(query)
    log_results_to_html("Display Resolutions", query, actual_output)

def test_camera_features():
    query = "What camera capabilities does the Galaxy S22 Ultra offer?"
    actual_output = ask_llm(query)
    log_results_to_html("Camera Features", query, actual_output)

def test_s_pen_functionality():
    query = "What can I do with the S Pen on the Galaxy S22 Ultra?"
    actual_output = ask_llm(query)
    log_results_to_html("S Pen Functionality", query, actual_output)
