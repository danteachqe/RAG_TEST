import os
import openai
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval

# --- OpenAI setup ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")
client = openai.OpenAI(api_key=api_key)

class GPT4ChatModel:
    def __init__(self):
        self.model_name = "gpt-4"
        self.client = client
        self.temperature = 0.7

    def generate(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"An error occurred: {e}"

gpt4 = GPT4ChatModel()

# --- Utilities ---
def load_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

def summarize_to_foreign(text, target_lang):
    messages = [{
        "role": "user",
        "content": f"Summarize the following English text in {target_lang}. Be concise and accurate.\n\n{text}"
    }]
    return gpt4.generate(messages)

def back_translate_to_english(foreign_text, from_language):
    messages = [{
        "role": "user",
        "content": f"Translate the following {from_language} text back into English:\n\n{foreign_text}"
    }]
    return gpt4.generate(messages)

# --- DeepEval Integration ---
def run_deepeval_metrics(input_text, ai_output):
    # Define test case
    test_case = LLMTestCase(
        input=input_text,
        actual_output=ai_output,
        expected_output=None  # GEval only needs input and actual_output
    )

    dataset = EvaluationDataset(test_cases=[test_case])

    # Define GEval metrics
    fluency = GEval(
        name="Fluency",
        criteria="Is the output grammatically correct and easy to understand?",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
    )
    coherence = GEval(
        name="Coherence",
        criteria="Is the output logically structured and cohesive?",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
    )
    relevance = GEval(
        name="Relevance",
        criteria="Does the output appropriately and directly answer the input?",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
    )
    concision = GEval(
        name="Concision",
        criteria="Is the output concise, avoiding redundancy and unnecessary verbosity while preserving meaning?",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
        
    )

    # Run DeepEval
    print("\n🔎 Evaluating with DeepEval (LLM-as-a-Judge)...")
    evaluate(
        dataset,
        metrics=[fluency, coherence, relevance, concision],
        skip_on_missing_params=True
    )

# --- Main Pipeline ---
def main():
    INPUT_FILE = "input_text.txt"
    REFERENCE_FILE = "reference_summary.txt"
    TARGET_LANGUAGE = "Romanian"

    print("🔹 Loading input and reference text...")
    source = load_text(INPUT_FILE)
    reference = load_text(REFERENCE_FILE)

    print("🌍 Generating summary in foreign language...")
    summary_foreign = summarize_to_foreign(source, TARGET_LANGUAGE)
    print(f"\n🌐 Foreign-Language Summary:\n{'-'*50}\n{summary_foreign}\n")

    print("🔁 Back-translating to English...")
    backtranslated_summary = back_translate_to_english(summary_foreign, TARGET_LANGUAGE)
    print(f"\n📝 Back-Translated Summary:\n{'-'*50}\n{backtranslated_summary}\n")

    print("📋 Reference Summary (English):\n" + "-"*50)
    print(reference + "\n")

    # Run DeepEval metrics
    run_deepeval_metrics(source, backtranslated_summary)

if __name__ == "__main__":
    main()
