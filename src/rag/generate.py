"""Prompt loading (versioned) + building the prompt from chunks + calling the LLM."""

from functools import lru_cache

import yaml

from rag.config import PROMPTS_DIR, load_config
from rag.guardrails import CHUNK_ID_RE
from rag.llm import get_llm

CITATION_FIELDS = ("chunk_id", "company", "ticker", "fiscal_year", "section", "page_start", "page_end", "source_url", "text")


@lru_cache
def load_prompt(version: str | None = None) -> dict:
    """Prompts live in configs/prompts/<version>.yaml with version/description/system/template."""
    version = version or load_config()["prompt"]["version"]
    with open(PROMPTS_DIR / f"{version}.yaml") as f:
        prompt = yaml.safe_load(f)
    assert prompt["version"] == version, f"{version}.yaml declares version {prompt['version']!r}"
    return prompt


def format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        pages = f"p.{c['page_start']}" if c["page_start"] == c["page_end"] else f"pp.{c['page_start']}-{c['page_end']}"
        header = f"[{c['chunk_id']}] {c['company']} ({c['ticker']}) | FY{c['fiscal_year']} | {c['section']} | {pages}"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


@lru_cache
def _llm():
    return get_llm()


def generate_answer(question: str, chunks: list[dict], prompt_version: str | None = None) -> dict:
    prompt = load_prompt(prompt_version)
    llm = _llm()
    answer = llm.generate(prompt["system"], prompt["template"].format(context=format_context(chunks), question=question))
    return {"answer": answer, "prompt_version": prompt["version"], "model": llm.name}


def extract_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """Structured citations for every retrieved chunk id that appears in the answer."""
    by_id = {c["chunk_id"]: c for c in chunks}
    ids = dict.fromkeys(CHUNK_ID_RE.findall(answer))
    return [{k: by_id[i][k] for k in CITATION_FIELDS} for i in ids if i in by_id]
