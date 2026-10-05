"""
Thin client for any OpenAI-compatible chat API (Featherless, OpenAI, Groq, etc.).

Config comes from environment variables (or Streamlit secrets, copied in by app.py):
  LLM_API_KEY   your API key                       (required)
  LLM_BASE_URL  default https://api.featherless.ai/v1
  LLM_MODEL     default Qwen/Qwen2.5-72B-Instruct  (any instruct model works)

Every response is cached on disk (cache/), keyed by a hash of the request,
so re-running on the same invoices is instant and costs nothing.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests

CACHE_DIR = Path(__file__).parent / "cache"
DEFAULT_BASE_URL = "https://api.featherless.ai/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"


class LLMError(RuntimeError):
    pass


def load_env(path=Path(__file__).parent / ".env"):
    """Minimal .env reader (KEY=value lines) so no extra package is needed."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def config():
    return {
        "api_key": os.getenv("LLM_API_KEY", "").strip(),
        "base_url": os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        "model": os.getenv("LLM_MODEL", DEFAULT_MODEL),
    }


def is_configured():
    return bool(config()["api_key"])


def cached(messages):
    """Return a cached answer for these messages if one exists (any model)."""
    raw = json.dumps({"m": messages}, sort_keys=True)
    key = hashlib.sha256(raw.encode()).hexdigest()[:32]
    p = CACHE_DIR / f"{key}.json"
    return (json.loads(p.read_text()) if p.exists() else None), p


def parse_json(text):
    """Pull the first JSON object out of a model reply (handles ```json fences)."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


def chat_json(messages, max_tokens=2000, retries=2):
    """Send a chat request and return the parsed JSON object from the reply.

    Results are cached by message content so the shipped sample cache works
    with or without an API key.
    """
    hit, path = cached(messages)
    if hit is not None:
        return hit

    cfg = config()
    if not cfg["api_key"]:
        raise LLMError("No LLM_API_KEY set. Add it to .env (local) or Streamlit secrets (deployed).")

    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(
                f"{cfg['base_url']}/chat/completions",
                headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
                json={"model": cfg["model"], "messages": messages, "temperature": 0, "max_tokens": max_tokens},
                timeout=120,
            )
            if r.status_code == 429 or r.status_code >= 500:
                raise LLMError(f"API busy ({r.status_code})")
            if r.status_code != 200:
                raise LLMError(f"API error {r.status_code}: {r.text[:300]}")
            content = r.json()["choices"][0]["message"]["content"]
            data = parse_json(content)
            CACHE_DIR.mkdir(exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
            return data
        except (requests.RequestException, LLMError, ValueError, KeyError) as e:
            last_err = e
            if isinstance(e, LLMError) and "API error" in str(e):
                break  # bad key / bad model: retrying won't help
            time.sleep(2 * (attempt + 1))
    raise LLMError(f"LLM request failed: {last_err}")
