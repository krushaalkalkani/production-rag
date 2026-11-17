"""LLM provider abstraction: the rest of the code calls `get_llm().generate(...)`
and never cares whether the model is Claude (API) or a local Ollama model.
"""

from rag.config import load_config


class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int, effort: str = "low"):
        import anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self.model, self.max_tokens, self.effort = model, max_tokens, effort
        self.name = f"anthropic:{model}"

    def generate(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
        )
        if resp.stop_reason == "refusal":  # the API's own safety classifier declined
            return "I can't answer that from the documents."
        return "".join(block.text for block in resp.content if block.type == "text").strip()


class OllamaLLM:
    def __init__(self, model: str, host: str):
        self.model, self.host = model, host.rstrip("/")
        self.name = f"ollama:{model}"

    def generate(self, system: str, user: str) -> str:
        import requests

        resp = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": 0},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


def get_llm():
    cfg = load_config()["llm"]
    if cfg["provider"] == "anthropic":
        return AnthropicLLM(**cfg["anthropic"])
    if cfg["provider"] == "ollama":
        return OllamaLLM(**cfg["ollama"])
    raise ValueError(f"Unknown llm.provider: {cfg['provider']!r} (use 'anthropic' or 'ollama')")
