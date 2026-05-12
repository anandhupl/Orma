"""
ORMA Evaluation Script — RAGAS Faithfulness & Answer Relevance

Student: Gemini 3.1 Flash Lite (same SDK call as api.py, context from eval dataset)
Judge:   GLM 5.1 on Modal (OpenAI-compatible critic_llm)

Dependencies: pip install ragas langchain-openai langchain-google-genai
Env vars:     GEMINI_API_KEY, MODAL_API_KEY
"""

import os
import json
import csv
import asyncio
import random

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

# ── Config ───────────────────────────────────────────────────────────────────
ORMA_MODEL   = "gemini-2.5-flash-lite"
GLM_BASE_URL  = "https://api.us-west-2.modal.direct/v1"
GLM_MODEL     = "glm-5.1"                # adjust if Modal deployment name differs
DATASET_PATH  = "eval_dataset_eng.json"
OUTPUT_PATH   = "eval_results.csv"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODAL_API_KEY  = os.getenv("MODAL_API_KEY")

# ── Student: Gemini direct SDK call ───────────────────────────────────────────
genai_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options={"api_version": "v1beta"},
)

SYSTEM_INSTRUCTION = """You are Orma, an AI Tutor for Kerala SSLC Chemistry.
You ground every answer strictly in the TEXTBOOK CONTEXT provided below.

LANGUAGE RULE:
Respond strictly in clear, easy-to-understand English.

MATH FORMATTING (Strict Protocol — no exceptions):
- Use \\(...\\) for inline math. Example: \\(V = IR\\)
- Use \\[...\\] for display equations. Example: \\[F = ma\\]
- NEVER use $ or $$ for math. They are forbidden.
- NEVER wrap math delimiters in markdown bolding (**). Output raw LaTeX only.

RESPONSE TEMPLATE (follow this structure for every answer):
1. Clear explanation of the topic in simple language.
---

2. 📝 **For the Exam:** The core definition or key point the student must remember.

---

3. 📖 **Reference:** Page [X] (cite the page number from the textbook context below).

--- TEXTBOOK CONTEXT ---
{context}
--- END TEXTBOOK CONTEXT ---
"""


async def get_student_answer(question: str, context: str) -> str:
    """Call Gemini directly with the eval question + its retrieved context.
    Retries on transient 503s with exponential backoff + jitter."""
    max_retries = 5
    base_delay = 4  # seconds
    for attempt in range(1, max_retries + 1):
        try:
            response = await genai_client.aio.models.generate_content(
                model=ORMA_MODEL,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=question)])],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION.format(context=context),
                ),
            )
            return response.text
        except Exception as e:
            is_retryable = "503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e)
            if not is_retryable or attempt == max_retries:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 2)
            print(f"    Retry {attempt}/{max_retries} after {delay:.1f}s — {e}")
            await asyncio.sleep(delay)


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    # 1. Load eval dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} questions from {DATASET_PATH}")

    # 2. Generate student answers via Gemini
    print("Generating student answers via Gemini...")
    student_answers = []
    for i, item in enumerate(raw_data):
        answer = await get_student_answer(item["question"], item["context"])
        student_answers.append(answer)
        print(f"  [{i+1}/{len(raw_data)}] Done")

    # 3. Build RAGAS dataset
    samples = []
    for item, answer in zip(raw_data, student_answers):
        samples.append(SingleTurnSample(
            user_input=item["question"],
            response=answer,
            reference=item["ground_truth"],
            retrieved_contexts=[item["context"]],
        ))
    dataset = EvaluationDataset(samples=samples)

    # 4. Judge LLM — GLM 5.1 on Modal (critic_llm)
    critic_llm = ChatOpenAI(
        base_url=GLM_BASE_URL,
        model=GLM_MODEL,
        api_key=MODAL_API_KEY,
    )

    # Embeddings for AnswerRelevance (reuse existing Gemini embedding model)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview",
        google_api_key=GEMINI_API_KEY,
    )

    # 5. Run RAGAS evaluation
    print("Running RAGAS evaluation (Faithfulness + Answer Relevance)...")
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy()],
        llm=critic_llm,
        embeddings=embeddings,
    )

    # 6. Write per-question results to CSV
    df = result.to_pandas()
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Question", "Ground Truth", "Student Answer",
            "Faithfulness Score", "Relevance Score",
        ])
        for i in range(len(raw_data)):
            writer.writerow([
                raw_data[i]["question"],
                raw_data[i]["ground_truth"],
                student_answers[i],
                df.iloc[i].get("faithfulness", "N/A"),
                df.iloc[i].get("answer_relevancy", "N/A"),
            ])

    print(f"\nResults saved to {OUTPUT_PATH}")
    print(f"Mean Faithfulness:   {df['faithfulness'].mean():.3f}")
    print(f"Mean Relevance:      {df['answer_relevancy'].mean():.3f}")


if __name__ == "__main__":
    asyncio.run(main())