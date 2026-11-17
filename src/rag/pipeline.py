"""The full question -> answer flow, shared by the CLI, the API and the eval scripts.

    retrieve -> (retrieval gate) -> generate -> (citation enforcement) -> log
"""

import json
import time
from datetime import datetime, timezone

from rag.config import PROJECT_ROOT, load_config
from rag.generate import extract_citations, generate_answer, load_prompt
from rag.guardrails import REFUSAL, enforce_citations
from rag.index import embedding_text
from rag.retrieve import retrieve


def answer_question(
    question: str, mode: str | None = None, enforce: bool | None = None, prompt_version: str | None = None, log: bool = True
) -> dict:
    cfg = load_config()
    mode = mode or cfg["retrieval"]["mode"]
    guard_cfg = cfg["guardrails"]
    enforce = guard_cfg["enabled"] if enforce is None else enforce
    prompt_version = load_prompt(prompt_version)["version"]

    t0 = time.perf_counter()
    chunks = retrieve(question, mode)
    t1 = time.perf_counter()

    best = chunks[0].get("rerank_score") if chunks else None
    guard = None
    if enforce and (not chunks or (best is not None and best < guard_cfg["min_rerank_score"])):
        # Layer 1: nothing relevant retrieved -> refuse, skip the LLM call entirely.
        answer, model, refused, reason = REFUSAL, None, True, "low_retrieval_score"
    else:
        gen = generate_answer(question, chunks, prompt_version)
        answer, model = gen["answer"], gen["model"]
        guard = enforce_citations(answer, chunks, guard_cfg)  # always computed, for metrics
        if enforce:
            answer, refused, reason = guard["answer"], guard["refused"], guard["refusal_reason"]
        else:
            refused = answer.strip() == REFUSAL
            reason = "model_refused" if refused else None
    t2 = time.perf_counter()

    result = {
        "question": question,
        "answer": answer,
        "citations": [] if refused else extract_citations(answer, chunks),
        "refused": refused,
        "refusal_reason": reason,
        "prompt_version": prompt_version,
        "model": model,
        "retrieval_mode": mode,
        "guardrails": enforce,
        "latency_ms": round((t2 - t0) * 1000),
        "retrieval_ms": round((t1 - t0) * 1000),
        "best_rerank_score": best,
        "citation_coverage": guard["citation_coverage"] if guard else None,
        "sentence_checks": guard["checks"] if guard else [],
        # full text of what the LLM saw - needed by the evaluator (RAGAS), not logged
        "contexts": [embedding_text(c) for c in chunks],
        "retrieved": [
            {k: c.get(k) for k in ("chunk_id", "section", "vector_score", "bm25_score", "rrf_score", "rerank_score")}
            for c in chunks
        ],
    }
    if log:
        _log(result)
    return result


def _log(result: dict) -> None:
    """Append every answer to a JSONL log, so any answer can be traced to the
    exact prompt version, retrieval mode and chunks that produced it."""
    path = PROJECT_ROOT / load_config()["logging"]["answers_log"]
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **result}
    entry["citations"] = [c["chunk_id"] for c in result["citations"]]
    entry.pop("contexts", None)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
