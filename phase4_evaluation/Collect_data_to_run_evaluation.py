"""
ORMA Evaluation Script — GENERATOR ONLY
Student: Gemini 3.1 Flash Lite (same SDK call as api.py)
Judge:   NONE (Manual Web UI Review)

Dependencies: pip install google-genai python-dotenv
Env vars:     GEMINI_API_KEY
"""

import os
import json
import asyncio
import random

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

# ── Config ───────────────────────────────────────────────────────────────────
ORMA_MODEL   = "gemini-2.5-flash-lite"
DATASET_PATH  = "eval_dataset_eng.json"
OUTPUT_PATH   = "student_responses_for_review.json"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("CRITICAL: GEMINI_API_KEY not found in .env")

# ── Student: Gemini direct SDK call ───────────────────────────────────────────
genai_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options={"api_version": "v1beta"},
)

SYSTEM_INSTRUCTION = """You are Orma, an AI Tutor for Kerala SSLC {{request.subject}}.
You ground every answer strictly in the TEXTBOOK CONTEXT provided below.

LANGUAGE RULE:
{{lang_rule}}

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
{{context}}
--- END TEXTBOOK CONTEXT ---
"""

async def get_student_answer(question: str, context: str) -> str:
    """Call Gemini directly with the eval question + its retrieved context.
    Retries on transient 503s with exponential backoff + jitter."""
    max_retries = 6 # Bumped up slightly for current high-demand spikes
    base_delay = 4  # seconds
    
    for attempt in range(1, max_retries + 1):
        try:
            response = await genai_client.aio.models.generate_content(
                model=ORMA_MODEL,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=question)])],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION.format(context=context),
                    temperature=0.2, # Low temp so the student doesn't hallucinate during eval
                ),
            )
            return response.text
        except Exception as e:
            is_retryable = "503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e)
            if not is_retryable or attempt == max_retries:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 2)
            print(f"    [!] Demand Spike: Retry {attempt}/{max_retries} after {delay:.1f}s — {e}")
            await asyncio.sleep(delay)

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    # 1. Load eval dataset
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found. Ensure the Phase 1 generation completed.")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} questions from {DATASET_PATH}")

    # 2. Generate student answers via Gemini
    print(f"Generating student answers via {ORMA_MODEL}...")
    results = []
    
    for i, item in enumerate(raw_data):
        print(f"  Generating answer [{i+1}/{len(raw_data)}]...")
        answer = await get_student_answer(item["question"], item["context"])
        
        # Build the exact JSON structure needed for the web UI Judge
        results.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "student_answer": answer
        })

    # 3. Save to JSON
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ SUCCESS! {len(results)} responses saved to {OUTPUT_PATH}")
    print("Next step: Upload student_responses_for_review.json to the chat for the Pro Evaluation.")

if __name__ == "__main__":
    asyncio.run(main())