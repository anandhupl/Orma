# Phase 1: Data Pipeline

This subsystem is responsible for the ingestion, OCR-based extraction, semantic chunking, and vector embedding of educational PDF materials. It operates strictly on the local file system to generate intermediate text artifacts, which are subsequently ingested directly into a Supabase vector database. Supabase acts as the permanent, exclusive data store for the downstream RAG pipeline.

## Architecture & Workflows

### 1. [Vision-OCR Extraction](./PDF-to-text-convert.py)
Converts raw PDF pages into structured Markdown utilizing multimodal LLM processing.

* **Image Processing:** Uses `PyMuPDF` (`fitz`) to render PDF pages into JPEG format, scaling via a `Matrix(2, 2)` transformation (144 DPI) to ensure legibility of mathematical symbols.
* **Extraction Engine:** Gemini 2.5 Pro (`gemini-2.5-pro`) processes visual data.
* **Prompt Engineering:** The system mandates the model to inject explicit page delimiters (`--- Page [Number] ---`), translate mathematical formulas to LaTeX, output Markdown tables, and strip non-essential headers/footers.
* **Concurrency Model:** Executes via `ThreadPoolExecutor` (capped at 5 workers to mitigate API rate limits). Implements exponential backoff using the `tenacity` library to handle network/quota failures.
* **Storage:** Concatenates and writes the final Markdown corpus to a local `.txt` file. This file is an ephemeral bridge for database ingestion.

### 2. [Embedding & Database Ingestion](./Ingest_to_database.py)
Parses the intermediate local Markdown artifacts, generates vector embeddings, and populates the remote Supabase PostgreSQL instance.

* **Chunking Strategy:** Employs a custom regular expression-based chunker (`smart_chunker`). Splits text at sentence boundaries with a 1500-character limit and a 200-character semantic overlap. Contextual page numbers are tracked via the OCR-injected delimiters.
* **Embedding Engine:** Gemini Embedding 2 Preview (`gemini-embedding-2-preview`) generating 768-dimensional vectors.
* **REST API Bypass:** Bypasses standard Google Generative AI SDKs by constructing raw HTTP POST requests to the `generativelanguage.googleapis.com` REST endpoint. This enables efficient `batchEmbedContents` execution without SDK overhead.
* **Rate Limiting Mitigation:** Batches are restricted to 20 chunks. A hardcoded 30-second delay (`time.sleep(30)`) is enforced between network calls to strictly adhere to the 30,000 Tokens-Per-Minute (TPM) quota.
* **Database Synchronization:** Connects to Supabase to execute a clear-and-replace strategy (filtered by `subject` and `language` slugs) before executing bulk inserts of text content, metadata, and embedding vectors.

## Security & Environment Requirements

Execution requires the following environment variables defined in a local `.env` file. **Hardcoding credentials in the source code is strictly prohibited.**

Required `.env` keys:
* `GEMINI_API_KEY`: Google AI Studio API key (Required for both extraction and ingestion).
* `SUPABASE_URL`: Target Supabase instance URL.
* `SUPABASE_KEY`: Supabase service role key (Requires write permissions).