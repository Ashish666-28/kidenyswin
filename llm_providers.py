"""Server-only RAG chat adapters and model routing configuration.

No provider is contacted until invoke(). Keys stay in the backend environment.
Run ``python -m backend.llm_providers`` for the supported provider list.
"""

import argparse
import ipaddress
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx


PROVIDER_TIMEOUT = 35
REQUEST_BUDGET_SECONDS = 100
MAX_CHAT_MODELS = 5
DEFAULT_MODEL_CHAIN = (
    ("gemini", "gemini-3.8-flash"),
    ("groq", "openai/gpt-oss-120b"),
    ("mistral", "mistral-small-latest"),
    ("zai", "glm-4.7-flash"),
    ("cloudflare", "@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
)
DEFAULT_PROVIDER_ORDER = ",".join(f"{provider}:{model}" for provider, model in DEFAULT_MODEL_CHAIN)


class ProviderRefusal(Exception):
    """Do not route around a provider's explicit safety block."""


class ProviderModel:
    """Adapters accept the remaining request timeout from the router."""


def _response(content, finish_reason="STOP"):
    return SimpleNamespace(content=content, response_metadata={"finish_reason": finish_reason})


def _messages(messages):
    return [{"role": "user" if role == "human" else role, "content": text}
            for role, text in messages]


def _loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


def _base_url(value, api_key):
    url = urlsplit(value)
    if (not url.hostname or url.username is not None or url.password is not None
            or url.query or url.fragment or url.scheme not in {"http", "https"}):
        raise ValueError("Provider base URL must be an HTTP(S) origin/path without credentials, query or fragment.")
    if url.scheme != "https" and not _loopback(url.hostname):
        raise ValueError("Provider base URL requires HTTPS except for a local loopback server.")
    if not api_key and not _loopback(url.hostname):
        raise ValueError("A remote provider requires an API key.")
    return value.rstrip("/")


def _post(url, body, headers, timeout):
    # No redirects, tools, request logging, or automatic retry loops.
    with httpx.Client(timeout=httpx.Timeout(timeout, connect=min(10, timeout))) as client:
        response = client.post(url, headers=headers, json=body)
        if response.status_code in {400, 403}:
            try:
                error = response.json().get("error") or {}
                if isinstance(error, dict) and error.get("code") in {
                    "content_filter", "content_policy_violation", "safety_violation",
                }:
                    raise ProviderRefusal()
            except (ValueError, AttributeError):
                pass
        response.raise_for_status()
        data = response.json()
    if not isinstance(data, dict) or data.get("error"):
        raise ValueError("Invalid provider response")
    return data


class GeminiModel(ProviderModel):
    def __init__(self, api_key, model_name):
        self.api_key, self.model_name = api_key, model_name

    def invoke(self, messages, timeout=PROVIDER_TIMEOUT):
        # Lazy construction lets other providers survive a Gemini setup failure.
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(
            model=self.model_name, api_key=self.api_key, vertexai=False,
            max_tokens=4096, timeout=timeout,
            # This SDK counts total attempts; 1 disables automatic retries.
            max_retries=1,
        )
        return model.invoke(messages)


class HTTPChatModel(ProviderModel):
    def __init__(self, api_key, model_name, base_url, token_parameter="max_tokens", extra_body=None):
        self.api_key, self.model_name = api_key, model_name
        self.base_url = _base_url(base_url, api_key)
        if token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("Unsupported token limit parameter")
        self.token_parameter = token_parameter
        self.extra_body = extra_body or {}

    def invoke(self, messages, timeout=PROVIDER_TIMEOUT):
        body = {"model": self.model_name, "messages": _messages(messages),
                self.token_parameter: 2048, "stream": False, **self.extra_body}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = _post(self.base_url + "/chat/completions", body, headers, timeout)
        choice = data["choices"][0]
        message = choice["message"]
        if message.get("refusal") or choice.get("finish_reason") == "content_filter":
            raise ProviderRefusal()
        content = message.get("content")
        if isinstance(content, list):
            if any(isinstance(block, dict) and block.get("type") == "refusal" for block in content):
                raise ProviderRefusal()
            content = [block for block in content if isinstance(block, dict)
                       and block.get("type") == "text" and not block.get("thought")]
        return _response(content, choice.get("finish_reason", ""))


class GroqModel(HTTPChatModel):
    def __init__(self, api_key, model_name, base_url="https://api.groq.com/openai/v1"):
        extra = {"reasoning_effort": "low", "include_reasoning": False} if model_name.startswith("openai/gpt-oss") else {}
        super().__init__(api_key, model_name, base_url, "max_completion_tokens", extra)


class CohereModel(ProviderModel):
    def __init__(self, api_key, model_name, base_url="https://api.cohere.com/v2"):
        self.api_key, self.model_name = api_key, model_name
        self.base_url = _base_url(base_url, api_key)

    def invoke(self, messages, timeout=PROVIDER_TIMEOUT):
        data = _post(self.base_url + "/chat", {
            "model": self.model_name, "messages": _messages(messages), "max_tokens": 2048,
        }, {"Authorization": f"Bearer {self.api_key}"}, timeout)
        finish = str(data.get("finish_reason", "")).upper()
        if finish == "ERROR_TOXIC":
            raise ProviderRefusal()
        text = "\n".join(block["text"] for block in data["message"].get("content", [])
                         if block.get("type") == "text" and isinstance(block.get("text"), str))
        return _response(text, finish)


@dataclass(frozen=True)
class ProviderSpec:
    base_url: str = ""
    default_model: str = ""
    kind: str = "chat"
    token_parameter: str = "max_tokens"


# Model IDs remain configurable; gateways with changing catalogs require an
# explicit MODEL/MODELS value instead of a guessed or retired default.
PROVIDERS = {
    "gemini": ProviderSpec(default_model=DEFAULT_MODEL_CHAIN[0][1], kind="gemini"),
    "groq": ProviderSpec("https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "groq"),
    "cohere": ProviderSpec("https://api.cohere.com/v2", "command-a-03-2025", kind="cohere"),
    "mistral": ProviderSpec("https://api.mistral.ai/v1", "mistral-small-latest"),
    "openrouter": ProviderSpec("https://openrouter.ai/api/v1", "openrouter/free"),
    "nvidia": ProviderSpec("https://integrate.api.nvidia.com/v1", "meta/llama-3.3-70b-instruct"),
    "sambanova": ProviderSpec("https://api.sambanova.ai/v1", "Meta-Llama-3.3-70B-Instruct"),
    "zai": ProviderSpec("https://api.z.ai/api/paas/v4", "glm-4.7-flash"),
    "cloudflare": ProviderSpec(default_model="@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
    "ollama": ProviderSpec("http://127.0.0.1:11434/v1"),
    "custom": ProviderSpec(),
}


class MisconfiguredModel(ProviderModel):
    def __init__(self, model_name):
        self.model_name = model_name

    def invoke(self, messages, timeout=PROVIDER_TIMEOUT):
        raise ValueError("Check this provider's MODEL and BASE_URL backend settings.")


def _key(value):
    value = (value or "").strip()
    return "" if value.lower().startswith(("your_", "paste_", "replace_")) else value


def _adapter(name, spec, key, model_name):
    prefix = name.upper()
    base_url = (os.getenv(prefix + "_BASE_URL") or spec.base_url).strip()
    if name == "cloudflare" and not base_url:
        account = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        if re.fullmatch(r"[a-zA-Z0-9_-]+", account):
            base_url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1"
    constructors = {"gemini": GeminiModel, "cohere": CohereModel, "groq": GroqModel}
    try:
        if not model_name:
            return MisconfiguredModel("<set " + prefix + "_MODEL>")
        if name == "zai" and model_name not in {"glm-4.7-flash", "glm-4.5-flash"}:
            return MisconfiguredModel(model_name)
        if name == "openrouter" and model_name != "openrouter/free" and not model_name.endswith(":free"):
            return MisconfiguredModel(model_name)
        if name in {"ollama", "custom"} and not _loopback(urlsplit(base_url).hostname):
            return MisconfiguredModel(model_name)
        if spec.kind == "gemini":
            return GeminiModel(key, model_name)
        if spec.kind in constructors:
            return constructors[spec.kind](key, model_name, base_url)
        # Keep the free GLM backup within the interactive response budget.
        extra = {"thinking": {"type": "disabled"}} if name == "zai" else None
        return HTTPChatModel(key, model_name, base_url,
                             os.getenv(prefix + "_TOKEN_PARAMETER") or spec.token_parameter, extra)
    except ValueError:
        # Bad configuration for one route must not disable healthy routes.
        return MisconfiguredModel(model_name)


def distinct_provider_models(routes):
    """Keep the first model per provider, up to five independent API providers."""
    selected, seen = [], set()
    for route, adapter in routes:
        provider = route.partition(":")[0].lower()
        if provider in seen:
            continue
        seen.add(provider)
        selected.append((route, adapter))
        if len(selected) == MAX_CHAT_MODELS:
            break
    return selected


def configured_models():
    """Select one model per provider, at most five; skip missing credentials."""
    entries = [item.strip() for item in os.getenv("LLM_PROVIDER_ORDER", DEFAULT_PROVIDER_ORDER).split(",")]
    if not all(entries) or len(set(entries)) != len(entries):
        raise ValueError("LLM_PROVIDER_ORDER must contain unique provider or provider:model entries.")
    models, seen = [], set()
    for entry in entries:
        provider, separator, explicit_model = entry.partition(":")
        name = provider.lower()
        if name not in PROVIDERS or (separator and not explicit_model.strip()):
            raise ValueError("Unknown provider or empty model in LLM_PROVIDER_ORDER. See docs/llm-providers.md.")
        spec = PROVIDERS[name]
        prefix = name.upper()
        key = _key(os.getenv(prefix + "_API_KEY"))
        if name == "gemini":
            key = _key(os.getenv("GOOGLE_API_KEY")) or key
        elif name == "cloudflare":
            key = key or _key(os.getenv("CLOUDFLARE_API_TOKEN"))
        model_csv = os.getenv(prefix + "_MODELS", "").strip()
        model_single = (os.getenv(prefix + "_MODEL") or spec.default_model).strip()
        if separator:
            model_names = [explicit_model.strip()]
        elif model_csv:
            model_names = [item.strip() for item in model_csv.split(",") if item.strip()]
        else:
            model_names = [model_single]
        # Local inference is opt-in by model, never by default installation.
        if name in {"ollama", "custom"}:
            if not any(model_names):
                continue
        elif not key:
            continue
        for model_name in model_names:
            if len(models) >= MAX_CHAT_MODELS:
                break
            if name in seen:
                continue
            seen.add(name)
            route = f"{name}:{model_name}" if separator or len(model_names) > 1 else name
            models.append((route, _adapter(name, spec, key, model_name)))
    return distinct_provider_models(models)


def main():
    parser = argparse.ArgumentParser(description="List supported RAG providers and configured routes without contacting them.")
    parser.add_argument("--configured", action="store_true", help="Print enabled route/model names only; never print keys.")
    args = parser.parse_args()
    if args.configured:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
        try:
            for route, model in configured_models():
                print(route if ":" in route else f"{route}: {model.model_name}")
        except ValueError as error:
            parser.error(str(error))
    else:
        for name, spec in PROVIDERS.items():
            print(f"{name:14} {spec.default_model or '(set ' + name.upper() + '_MODEL)'}")


if __name__ == "__main__":
    main()
