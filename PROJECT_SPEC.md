# Project 1 — Production-Grade RAG with Citations, Evaluation and CI Gating

> **For Claude (the builder):** Build this project phase by phase. After each phase, stop, show how to run it, and explain what was built in simple language — the owner (Krushal) is learning ML engineering through this project and must be able to explain every part in an interview. Prefer simple, readable code over clever code. Never skip the evaluation pieces; they are the point of the project.

## 1. What we are building
A domain-specific **"Ask My Docs"** system: a user asks a question about a document collection and gets an answer **with citations pointing to the exact source passage**. If the documents don't support an answer, the system **refuses** instead of guessing.

The difference from a tutorial: hybrid retrieval, re-ranking, citation enforcement, versioned prompts, a golden evaluation set, and a CI pipeline that **fails the build if answer quality drops**.

**Domain (default): SEC 10-K annual reports** of ~10–20 public companies (e.g. Apple, Microsoft, NVIDIA, Tesla, Amazon). Why: public domain, long and messy PDFs/HTML, real business value ("What were NVIDIA's main risk factors in 2025?"), and it fits the owner's interest in finance/trading. Download from SEC EDGAR (https://www.sec.gov/edgar). Respect EDGAR's fair-access rules (set a descriptive User-Agent, stay under 10 requests/second).

## 2. Why it matters (for the resume)
RAG is the most common enterprise AI pattern. The gap between a RAG demo and a production RAG system is huge — this project lives in that gap.

## 3. Tech stack
| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | `uv` for env/deps |
| Orchestration | LangChain (or LangGraph) | keep usage light; plain Python is fine where clearer |
| Parsing | `pypdf` / `unstructured` / BeautifulSoup for HTML | |
| Embeddings | `sentence-transformers` (`BAAI/bge-small-en-v1.5` default) | free, local |
| Vector store | ChromaDB (local, persistent) | Weaviate is the alternative |
| Keyword search | `rank_bm25` | |
| Re-ranker | cross-encoder `BAAI/bge-reranker-base` or `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cohere Rerank optional |
| LLM | Configurable: Ollama local (free, e.g. `llama3.2:3b` / `qwen3:8b`) **or** Anthropic API (`claude-haiku-5-5`) | provider selected in config |
| API | FastAPI | |
| UI | Streamlit or Gradio (simple) | |
| Evaluation | RAGAS + custom scripts | |
| CI | GitHub Actions | |
| Tests | pytest | |

## 4. Architecture
```
            ┌──────────────── Ingestion (offline) ────────────────┐
 10-K docs ─► parse ─► clean ─► chunk (500–800 tok, ~100 overlap) ─► embed ─► Chroma
                                       └──────────────────────────► BM25 index
            └──────────────────────────────────────────────────────┘

 Query ─► BM25 top-k ─┐
        └► Vector top-k┴► merge (RRF) ─► cross-encoder re-rank ─► top-n chunks
                                                                    │
                       support check ◄──────────────────────────────┘
                          │ not supported → "I can't answer that from the documents."
                          ▼ supported
                 LLM (versioned prompt) ─► answer + [citations: doc, section, page, chunk_id]
```

## 5. Suggested repo structure
```
production-rag/
├── README.md                 # results table, diagram, demo
├── pyproject.toml
├── configs/
│   ├── config.yaml           # chunk size, top_k, models, thresholds
│   └── prompts/
│       ├── answer_v1.yaml    # versioned prompts
│       └── answer_v2.yaml
├── data/
│   ├── raw/                  # downloaded filings (gitignored)
│   └── processed/
├── src/rag/
│   ├── ingest.py             # download + parse + clean
│   ├── chunking.py
│   ├── index.py              # embeddings + Chroma + BM25
│   ├── retrieve.py           # hybrid + RRF
│   ├── rerank.py
│   ├── generate.py           # prompt loading, LLM call, citation formatting
│   ├── guardrails.py         # citation enforcement / refusal
│   ├── llm.py                # provider abstraction (ollama | anthropic)
│   └── api.py                # FastAPI
├── app/ui.py                 # Streamlit/Gradio
├── eval/
│   ├── golden_set.jsonl      # 50–200 hand-verified Q&A pairs
│   ├── run_eval.py
│   └── reports/
├── tests/
└── .github/workflows/eval.yml
```

## 6. Phases

### Phase 1 — Fundamentals: working RAG with citations
Tasks:
1. Download 10-K filings for ~10 companies (script, not manual).
2. Parse PDF/HTML to text; keep metadata: company, year, section (e.g. "Item 1A Risk Factors"), page.
3. Chunk to 500–800 tokens with ~100-token overlap (overlap prevents cutting a key sentence in half). Each chunk gets a stable `chunk_id`.
4. Embed chunks, store in persistent Chroma with metadata.
5. Retrieve top-k by vector similarity; generate an answer that cites `[company, year, section, chunk_id]`.
6. CLI: `python -m rag.ask "What are Tesla's main risk factors?"`

Acceptance criteria:
- Ask a question → get an answer + citations → open the cited chunk and confirm the answer is actually there.
- Ingestion is re-runnable and idempotent.

### Phase 2 — Production quality
Tasks:
1. **Hybrid retrieval:** BM25 + vector search, merged with Reciprocal Rank Fusion. (Vector = meaning; BM25 = exact terms like "EBITDA" or a product name.)
2. **Cross-encoder re-ranker:** re-score the merged top ~20 as (query, chunk) pairs; keep top ~5.
3. **Citation enforcement:** if the best re-rank score is below a threshold, or the answer's claims aren't supported by retrieved chunks, return a refusal. Every sentence in an answer must map to at least one citation.
4. **Versioned prompts** in `configs/prompts/*.yaml` (with `version`, `description`, `template`); the active version is set in `config.yaml` and logged with every answer.
5. FastAPI endpoint `POST /ask` → `{answer, citations[], refused, prompt_version, latency_ms}`.
6. Simple UI showing the answer with clickable citations that reveal the source text.

Acceptance criteria:
- An ablation table comparing: vector-only vs hybrid vs hybrid+rerank (on the golden set from Phase 3 — build a small version early).
- Questions outside the corpus ("What is Apple's 2030 revenue?") are refused.

### Phase 3 — Shippable: evaluation + CI
Tasks:
1. **Golden set:** 50–200 Q&A pairs, manually verified, stored in `eval/golden_set.jsonl` with `question`, `reference_answer`, `source_chunk_ids`, and a few `should_refuse: true` cases.
2. `eval/run_eval.py` measures:
   - **Faithfulness** (are the answer's claims supported by retrieved chunks?) — RAGAS
   - **Answer relevance / correctness** vs reference — RAGAS
   - **Context recall / retrieval hit rate@k** (was a correct source chunk retrieved?)
   - **Citation coverage** (% of answer sentences with a valid citation)
   - **Refusal accuracy** (refused when it should, answered when it should)
3. Save a JSON + Markdown report per run in `eval/reports/`.
4. **GitHub Actions** workflow on every pull request: build index on a small fixture corpus, run eval, **fail if faithfulness or retrieval hit rate drops below thresholds** in `config.yaml`.

Acceptance criteria:
- A PR that deliberately breaks the prompt causes CI to fail.
- README contains the final metrics table.

## 7. Results table to put in README (fill with real numbers)
| Setup | Hit rate@5 | Faithfulness | Answer correctness | Refusal accuracy | p50 latency |
|---|---|---|---|---|---|
| Vector only | | | | | |
| Hybrid (BM25 + vector) | | | | | |
| Hybrid + re-rank | | | | | |
| + citation enforcement | | | | | |

## 8. Definition of done
- [ ] One-command setup (`make setup && make ingest && make serve`)
- [ ] Answers cite exact source passages; unsupported questions are refused
- [ ] Hybrid retrieval + re-ranker with measured improvement
- [ ] Versioned prompts
- [ ] Golden set + eval script + CI gate that actually fails on regressions
- [ ] README: problem, architecture diagram, results table, what went wrong & how I fixed it, demo GIF

## 9. Resume bullet template (fill with real numbers)
- Built a production RAG system over 10-K filings with hybrid BM25+vector retrieval and cross-encoder re-ranking, improving retrieval hit rate@5 from **X%** to **Y%** and faithfulness from **A** to **B** on a 150-question hand-verified golden set.
- Added citation enforcement and refusal logic, reducing unsupported answers to **Z%**; gated every PR with automated RAGAS evaluation in GitHub Actions.

## 10. Interview questions this project prepares you for
- Why chunk overlap? How did you pick chunk size?
- Why hybrid retrieval? When does BM25 beat embeddings?
- What does a cross-encoder do differently from a bi-encoder?
- How do you measure hallucination / faithfulness?
- How do you stop quality regressions in an LLM system?
