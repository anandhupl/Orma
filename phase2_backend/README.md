# Phase 2: Backend Services

This subsystem operates as the orchestration layer for the Orma application. Built on FastAPI, it manages client routing, RAG retrieval workflows, rate limiting, and streaming inference using the Gemini 2.5 Flash-Lite model.

## Architecture & Workflows

### 1. Client Routing & Delivery ([GET /](./api.py))
The root endpoint acts as a reverse proxy for static frontend assets. 
* Executes User-Agent string evaluation to detect mobile environments. 
* Dynamically serves `Mobile_code.html` or `Desktop_code.html`. Returns a 500 Internal Server Error if the target artifact is missing.

### 2. Retrieval-Augmented Generation (RAG) Pipeline
The RAG workflow is executed asynchronously within the [get_rag_context_async](./api.py) function.

* **Query Vectorization:** Transforms the incoming user query into a 768-dimensional vector via `gemini-embedding-2-preview`.
* **In-Memory Caching:** Implements a localized, dictionary-based LRU-style cache (`_rag_context_cache`, max 200 items) keyed by the hash of `query:subject:language`. This prevents redundant Supabase RPC calls but is isolated per-process in multi-worker environments.
* **Database Query (REST Execution):** Bypasses the Supabase Python SDK to construct raw HTTP POST requests via `httpx` to the `hybrid_search_v2` RPC. Implements a dynamic `match_count` strategy: strictly limited to 1 for short queries (<5 words) and 2 for longer academic inquiries.
* **Context Assembly:** Parses the database JSON payload, prepends page metadata, concatenates text blocks, and enforces a hard truncation limit of 3000 characters to optimize context window efficiency.

### 3. Inference Engine ([POST /ask-orma](./api.py))
The primary endpoint handling stateful conversation logic and LLM stream generation.

* **Rate Limiting:** Employs SlowAPI restricted to 30 requests per minute. Limits are keyed dynamically by a custom HTTP header (`X-Session-ID`) acting as a failover to IP addresses, ensuring functionality behind shared school NAT gateways.
* **Intent Routing:** Evaluates query length and string matching to bypass the RAG pipeline for casual conversational inputs (e.g., "hi", "thanks").
* **Prompt Engineering & Constraint Injection:** * Injects strict formatting rules for `Physics`, `Chemistry`, and `Mathematics` to force linear LaTeX formatting (e.g., preventing unsupported vertical `\begin{array}` structures).
    * Encloses user queries within explicit `<student_query>` tags to define rigid system boundaries.
* **Security & Input Destruction:** Implements a regex-based character deletion strategy (`re.sub`) that removes all XML-style angle brackets. **Note:** This neutralizes prompt injection but destroys mathematical inequalities (e.g., < or > symbols) in student queries.
* **Context Management:** Maintains a sliding window of the last 5 turns. Historical strings are aggressively truncated (User: 200 chars, AI: 150 chars) to maintain minimal latency at the cost of long-term semantic continuity.
* **Streaming Delivery:** Manages Server-Sent Events (SSE) data streams via FastAPI's `StreamingResponse`. Utilizes the `tenacity` library to provide exponential backoff and localized error handling for API connection timeouts or model unavailability.

## Dependencies & Environment

The backend requires the following configuration parameters defined in the Hugging Face Secrets UI or a local `.env` file:

* `GEMINI_API_KEY`: Required for embeddings and inference.
* `SUPABASE_URL`: Required for the REST RPC execution.
* `SUPABASE_KEY`: Required for database authorization.
* `PORT`: Port binding for Uvicorn. Defaults to 8000.
* `FRONTEND_URL`: Defines the authorized origin for CORS middleware.