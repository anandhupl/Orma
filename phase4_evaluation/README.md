**Phase 4: Evaluation Framework**


This subsystem governs the quality assurance and performance benchmarking of the Orma pipeline. It utilizes an LLM-as-a-Judge architecture to quantify hallucination rates and contextual accuracy. 

**Architectural Note:** Currently, this evaluation framework operates in isolation. It benchmarks the inference capability of the model using static, pre-compiled datasets. It does *not* execute End-to-End (E2E) testing against the live FastAPI backend, the local dictionary cache, or the Supabase retrieval pipeline.

**Architecture & Workflows**


### 1. [Automated Ragas Evaluation](./run_evaluation.py)
Executes a fully automated, quantitative evaluation using the Ragas framework by directly executing Google GenAI SDK calls.

* **Student Model:** Gemini 2.5 Flash-Lite. It replicates the system prompt formatting constraints defined in the Phase 2 backend, injecting context directly from a local `eval_dataset_eng.json` file.
* **Critic/Judge Model:** GLM 5.1 deployed via Modal Serverless Infrastructure. It interfaces through an OpenAI-compatible client routing to `https://api.us-west-2.modal.direct/v1`.
* **Embedding Engine:** `gemini-embedding-2-preview` (via LangChain's `GoogleGenerativeAIEmbeddings`) handles the vector comparisons required for relevance scoring.
* **Evaluation Metrics:**
  * **Faithfulness:** Measures if the generated answer is strictly grounded in the provided static textbook context.
  * **Answer Relevancy:** Measures how directly the generated answer addresses the original query.
* **Artifact Output:** Generates `eval_results.csv` containing question, ground truth, retrieved context, student answer, and the calculated metric scores.

### 2. [Manual Review Generation (Accessible Track)](./generate_responses_for_review.py)
A lightweight pipeline designed for Human-In-The-Loop (HITL) manual inspection and for developers who want to test the system without configuring Modal infrastructure.

* **Execution:** Iterates over the static `eval_dataset_eng.json` dataset, executing direct SDK calls to the Gemini API to generate student responses. **This bypasses the GLM 5.1 critic, meaning it only requires a Gemini API key to run.**
* **Artifact Output:** Dumps a structured JSON file (`student_responses_for_review.json`) that can be read manually or ingested into front-end review tools.

**Environment & Dependencies**


Execution requires the following configuration parameters inside a `.env` file located in the root directory.

Required `.env` keys:
* `GEMINI_API_KEY`: Required for student model inference and Ragas embedding extraction.
* `MODAL_API_KEY`: Required to authenticate the GLM 5.1 critic endpoint during automated evaluation.

Required Python Libraries:
* `ragas`
* `langchain-openai`
* `langchain-google-genai`
* `google-genai`