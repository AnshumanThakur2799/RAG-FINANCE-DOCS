# RAG Finance Docs

Production-focused Retrieval-Augmented Generation (RAG) system for finance and tender document intelligence.  
The project ingests PDF documents, enriches them with structured metadata, performs dense/lexical/hybrid retrieval, and generates citation-grounded answers using an OpenAI-compatible LLM endpoint.

## Assumptions

The following assumptions are made to produce a complete, implementation-ready README:

- **Project name:** `RAG Finance Docs` (inferred from repository name).
- **Project type:** Enterprise RAG system for document Q&A and retrieval benchmarking.
- **Primary goal:** Reduce manual effort in extracting deadlines, owners, obligations, and key facts from finance/tender PDFs with auditable citations.
- **Target users:** Developers, analysts, and enterprise operations/procurement teams.
- **Deployment model:** Local-first development (Python CLI/UI), with optional integration to self-hosted RAGFlow APIs.
- **LLM model:** `meta-llama/Meta-Llama-3-8B-Instruct` via DeepInfra OpenAI-compatible endpoint (configurable).
- **Missing modules:** Scripts reference `app/db/*` modules (`sqlite_store`, `vector_store`), which are assumed to be part of the intended codebase.

## 1. Project Overview

`RAG Finance Docs` is a modular Python RAG pipeline designed for enterprise document intelligence.

- Ingests PDF files and optional CSV metadata.
- Splits documents into token-aware chunks.
- Builds embeddings via OpenAI, Azure OpenAI, or local Sentence Transformers.
- Stores vectors in LanceDB or Qdrant (provider configurable).
- Supports retrieval modes: `dense`, `lexical`, and `hybrid` (RRF fusion).
- Generates grounded answers with citation constraints.
- Provides benchmarking utilities for retrieval quality (`recall@k`, `precision@k`, `MRR@k`, `NDCG@k`).

## 2. Architecture

```text
                         +-----------------------------+
                         |      Metadata CSV (opt)     |
                         |   (tenders_data.csv, etc.)  |
                         +-------------+---------------+
                                       |
                                       v
+-------------------+         +------------------------+
|   PDF Documents   | ----->  |   Ingestion Pipeline   |
| (sample_data_pdf) |         | index_pdfs / async_*   |
+---------+---------+         +-----------+------------+
          |                               |
          | extract_text + chunking       | metadata attach
          v                               v
  +------------------+            +---------------------+
  | Token Chunker    |            | SQLite Doc Store    |
  | (tiktoken based) |            | docs + lexical FTS  |
  +---------+--------+            +----------+----------+
            |                                |
            | embeddings                     | lexical retrieval
            v                                v
  +-------------------------+      +--------------------------+
  | Embedding Client        |      | Lexical Retriever        |
  | OpenAI/Azure/Local      |      | (SQLite FTS)             |
  +------------+------------+      +------------+-------------+
               |                                |
               | dense retrieval                |
               v                                |
      +----------------------+                  |
      | Vector Store         |                  |
      | LanceDB / Qdrant     |                  |
      +----------+-----------+                  |
                 \                             /
                  \                           /
                   v                         v
                 +-----------------------------+
                 | Retriever (dense/lexical/   |
                 | hybrid with RRF fusion)     |
                 +--------------+--------------+
                                |
                                v
                 +-----------------------------+
                 | LLM Generation Layer        |
                 | DeepInfra OpenAI-compatible |
                 +--------------+--------------+
                                |
                                v
                 +-----------------------------+
                 | UI / CLI Output             |
                 | Gradio, Tkinter, scripts    |
                 +-----------------------------+
```

## 3. Prompt Design

The generation layer enforces grounded, enterprise-safe behavior:

- Answer only from retrieved context.
- Return a fallback statement when evidence is missing.
- Enforce inline citations in the form `[source#chunk_id]`.
- Suppress chain-of-thought and hidden reasoning tags.

### System Prompt Template

```text
You are an internal Enterprise AI Assistant for business teams.

Answer strictly from the provided document context.
Do not invent facts and do not use external knowledge.
If the answer is not explicitly supported in context, respond:
"Insufficient information in the provided documents."

Behavior rules:
1. Answer only what is asked.
2. Keep responses concise and decision-ready.
3. Cite factual statements using: [source#chunk_or_page].
4. Do not output chain-of-thought or internal reasoning.
```

### User Prompt Template

```text
Question:
{user_query}

Context:
[{source_a}#{chunk_a}] {chunk_text_a}
[{source_b}#{chunk_b}] {chunk_text_b}
...

Return only the direct answer to the question.
Rules:
- Include citation(s) for factual claims using [source#chunk_or_page].
- If info is missing, say "Insufficient information in the provided documents."
- No chain-of-thought, no XML tags.
```

## 4. Model & Retrieval Strategy

### Embeddings

- **Providers:** OpenAI, Azure OpenAI, or local HuggingFace/SentenceTransformer.
- **Default embedding model:** `text-embedding-3-small` (OpenAI).
- **Local option:** `intfloat/e5-small-v2` with prompt-style prefixes (`query:` / `passage:`).

### LLM

- **Provider:** DeepInfra (OpenAI-compatible API client).
- **Default model:** `meta-llama/Meta-Llama-3-8B-Instruct`.
- **Configurable controls:** `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`.

### Retrieval Modes

- **Dense:** semantic vector similarity from vector DB.
- **Lexical:** keyword/full-text matching via SQLite chunk index.
- **Hybrid (default):** Reciprocal Rank Fusion (RRF) of dense + lexical results.

Hybrid controls:

- `HYBRID_RRF_K` (fusion weighting parameter)
- `HYBRID_CANDIDATE_MULTIPLIER` (candidate expansion before fusion)

## 5. Guardrails & Safety Design

- **Grounding-only responses:** prompts instruct the model to avoid external knowledge.
- **Insufficient-evidence fallback:** standardized response for unsupported queries.
- **Citation enforcement:** answers are expected to include `[source#chunk]`; fallback citations are appended if missing.
- **Reasoning leak mitigation:** `<think>...</think>` content is removed before returning output.
- **Determinism controls:** low default temperature (`0.2`) and bounded output length.
- **Metadata-aware filtering:** retrieval supports structured filters (organization, tender id, date ranges).

## 6. Installation & Setup Instructions

### Prerequisites

- Python 3.10+ recommended
- `pip`
- Optional: running Qdrant instance if using `VECTOR_DB_PROVIDER=qdrant`

### Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell

pip install -r requirements.txt
cp .env.example .env
```

Populate `.env` with required credentials/settings:

- `OPENAI_API_KEY` (if `EMBEDDING_PROVIDER=openai`)
- `AZURE_OPENAI_*` fields (if `EMBEDDING_PROVIDER=azure`)
- `DEEPINFRA_API_KEY` for LLM generation
- Vector backend selection via `VECTOR_DB_PROVIDER`

### Index Documents

```bash
python -m app.ingestion.index_pdfs \
  --input sample_data_pdf \
  --metadata-csv sample_data_pdf/lcwa_gov_pdf_metadata.csv \
  --table tender_chunks
```

Async indexing for larger corpora:

```bash
python -m app.ingestion.async_index_pdfs \
  --input sample_data_pdf \
  --metadata-csv sample_data_pdf/lcwa_gov_pdf_metadata.csv \
  --table tender_chunks \
  --max-concurrency 4
```

### Query Retrieval

```bash
python -m app.query.search \
  --query "What is the bid submission end date?" \
  --top-k 5 \
  --retrieval-mode hybrid \
  --table tender_chunks
```

### Launch UI

```bash
python -m app.ui.embedding_gradio
# or
python -m app.ui.embedding_gui
```

## 7. Example Usage (API Call Example)

This repository includes scripts to integrate with a RAGFlow API instance.

### cURL: list datasets from RAGFlow

```bash
curl -X GET "http://localhost/api/v1/datasets?page=1&page_size=20" \
  -H "Authorization: Bearer <RAGFLOW_API_KEY>" \
  -H "Content-Type: application/json"
```

### Python: upload + metadata + parse using project script

```bash
python rag_flow_upload.py \
  --input sample_data_pdf \
  --metadata-csv sample_data_pdf/lcwa_gov_pdf_metadata.csv \
  --base-url http://localhost \
  --api-key <RAGFLOW_API_KEY> \
  --dataset-name tenders_data
```

## 8. Sample Output

```text
Question:
What is the bid submission end date for tender 2025_AB12_100_1?

Answer:
The bid submission end date is 2025-08-12T17:00:00. [2025_AB12_100_1/tender_notice.pdf#12]

It is listed under the tender timeline as the final submission deadline. [2025_AB12_100_1/tender_notice.pdf#13]
```

## 9. Configuration Options

Key environment variables from `.env.example`:

| Variable | Purpose | Default |
|---|---|---|
| `EMBEDDING_PROVIDER` | Embedding backend (`openai`, `azure`, `local`) | `openai` |
| `OPENAI_EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-small` |
| `LOCAL_EMBEDDING_MODEL` | Local SentenceTransformer model | `intfloat/e5-small-v2` |
| `LLM_PROVIDER` | LLM provider implementation | `deepinfra` |
| `DEEPINFRA_MODEL` | Generation model identifier | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `LLM_TEMPERATURE` | Generation temperature | `0.2` |
| `VECTOR_DB_PROVIDER` | Vector DB backend (`lancedb`, `qdrant`) | `lancedb` |
| `LANCEDB_DIR` | LanceDB path | `data/lancedb` |
| `QDRANT_URL` | Qdrant endpoint | `http://localhost:6333` |
| `CHUNK_SIZE_TOKENS` | Chunk size for ingestion | `500` |
| `CHUNK_OVERLAP_TOKENS` | Chunk overlap | `50` |
| `RETRIEVAL_MODE` | Default retrieval mode | `hybrid` |
| `HYBRID_RRF_K` | RRF constant | `60` |
| `HYBRID_CANDIDATE_MULTIPLIER` | Candidate expansion before fusion | `4` |

## 10. Limitations

- No built-in HTTP API service layer in this repo (CLI/UI oriented).
- Runtime depends on missing `app/db` modules in current snapshot.
- OCR quality depends on PDF text extractability (`markitdown[pdf]` behavior may vary by document).
- Answer quality depends on chunk quality and embedding model selection.
- Citation format is enforced heuristically and does not guarantee perfect source alignment in all cases.

## 11. Future Improvements

- Add first-class FastAPI service with `/index`, `/search`, `/ask` endpoints.
- Add reranking stage (cross-encoder) after first-pass retrieval.
- Add structured citation validator to verify each answer claim against chunk text.
- Add containerized deployment (`Dockerfile`, `docker-compose`, optional Kubernetes manifests).
- Add unit/integration tests for ingestion, retrieval fusion, and prompt guardrails.
- Add observability (latency, token, and retrieval-quality dashboards).

## 12. Folder Structure

```text
RAG-FINANCE-DOCS/
├── app/
│   ├── config.py
│   ├── embeddings/
│   │   └── client.py
│   ├── ingestion/
│   │   ├── index_pdfs.py
│   │   ├── async_index_pdfs.py
│   │   ├── chunker.py
│   │   └── pdf_reader.py
│   ├── retrieval/
│   │   └── pipeline.py
│   ├── query/
│   │   └── search.py
│   ├── llm/
│   │   └── client.py
│   ├── ui/
│   │   ├── embedding_gradio.py
│   │   └── embedding_gui.py
│   └── evaluation/
│       └── score_qna_accuracy.py
├── rag_flow_upload.py
├── rag_flow_delete_all.py
├── qna_data.csv
├── sample_data_pdf/
├── sample_pdf_v2/
├── .env.example
├── requirements.txt
└── README.md
```

## Quick Start Checklist

- Configure `.env` with embedding and LLM credentials.
- Index PDF corpus with `app.ingestion.index_pdfs` (or async variant).
- Validate retrieval quality with `app.evaluation.score_qna_accuracy`.
- Run Gradio/Tkinter UI for interactive Q&A with citations.

