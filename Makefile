.PHONY: setup ingest ask serve ui golden ablation eval ci-local test

setup:          ## install Python + all dependencies into .venv
	uv sync

ingest:         ## download 10-Ks, parse, chunk, embed, store in Chroma (safe to re-run)
	uv run python -m rag.ingest

ask:            ## usage: make ask Q="What are Tesla's main risk factors?"
	uv run python -m rag.ask "$(Q)" --show-sources

serve:          ## API on http://localhost:8000 (docs at /docs)
	uv run uvicorn rag.api:app --port 8000

ui:             ## Streamlit UI on http://localhost:8501 (needs `make serve` running)
	uv run streamlit run app/ui.py

golden:         ## rebuild eval/golden_set.jsonl from eval/golden_seed.yaml
	uv run python eval/build_golden.py

ablation:       ## vector vs hybrid vs hybrid+rerank (+guardrails) -> eval/reports/
	uv run python eval/ablation.py

test:
	uv run pytest -q

eval:           ## full evaluation on the golden set (RAGAS judge, ~$3.50)
	uv run python eval/run_eval.py

ci-local:       ## exactly what GitHub Actions runs: fixture index + gated eval (~$2)
	RAG_CHROMA_DIR=data/chroma_ci uv run python -m rag.ingest --from-processed data/fixtures
	RAG_CHROMA_DIR=data/chroma_ci uv run python eval/run_eval.py --tickers AAPL TSLA NVDA --name ci --gate
