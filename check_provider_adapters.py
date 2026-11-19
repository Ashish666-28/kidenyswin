"""Offline checks for configured RAG routes and provider wire formats.

Only synthetic keys, questions and replies are used. HTTP requests are handled
by MockTransport; these checks never contact a model provider or read .env.
"""

import json
import os
import unittest
from unittest.mock import patch

import httpx

from llm_providers import (
    PROVIDERS,
    CohereModel,
    HTTPChatModel,
    ProviderRefusal,
    configured_models,
    distinct_provider_models,
)


MESSAGES = [("system", "Use the supplied test passages."), ("human", "Test question")]


class ConfigurationChecks(unittest.TestCase):
    def configured(self, environment):
        with patch.dict(os.environ, environment, clear=True):
            return configured_models()

    def test_registry_exposes_supported_routes(self):
        expected = {
            "gemini", "groq", "cohere", "mistral", "openrouter", "nvidia",
            "sambanova", "zai", "cloudflare", "ollama", "custom",
        }
        self.assertEqual(expected, set(PROVIDERS))

    def test_no_keys_does_not_enable_cloud_or_local_models(self):
        self.assertEqual(self.configured({}), [])

    def test_existing_two_keys_enable_only_two_distinct_providers(self):
        routes = self.configured({"GOOGLE_API_KEY": "test-google-key", "GROQ_API_KEY": "test-groq-key"})
        self.assertEqual([label for label, _ in routes], [
            "gemini:gemini-3.8-flash", "groq:openai/gpt-oss-120b",
        ])
        self.assertTrue(all(type(model).__name__ != "MisconfiguredModel" for _, model in routes))

    def test_model_lists_select_only_one_model_per_provider(self):
        routes = self.configured({
            "LLM_PROVIDER_ORDER": "ollama,groq", "OLLAMA_MODELS": "first,second,first,third,fourth,fifth,sixth",
            "GROQ_API_KEY": "test-groq-key",
        })
        self.assertEqual([model.model_name for _, model in routes], ["first", "openai/gpt-oss-120b"])

    def test_five_distinct_provider_cap_applies_after_repeated_providers(self):
        routes = [("gemini:first", object()), ("gemini:second", object())]
        routes.extend((provider, object()) for provider in ["groq", "mistral", "zai", "cloudflare", "cohere"])
        self.assertEqual([name for name, _ in distinct_provider_models(routes)],
                         ["gemini:first", "groq", "mistral", "zai", "cloudflare"])

    def test_configuration_caps_distinct_providers_without_reusing_google(self):
        routes = self.configured({
            "LLM_PROVIDER_ORDER": "gemini:first,gemini:second,groq,mistral,zai,cloudflare,cohere",
            "GOOGLE_API_KEY": "test-google", "GROQ_API_KEY": "test-groq",
            "MISTRAL_API_KEY": "test-mistral", "ZAI_API_KEY": "test-zai",
            "CLOUDFLARE_API_TOKEN": "test-cf", "CLOUDFLARE_ACCOUNT_ID": "test-account",
            "COHERE_API_KEY": "test-cohere",
        })
        self.assertEqual([name for name, _ in routes], ["gemini:first", "groq", "mistral", "zai", "cloudflare"])

    def test_each_cloud_provider_can_be_enabled_independently(self):
        for provider in PROVIDERS:
            if provider in {"custom", "ollama"}:
                continue
            with self.subTest(provider=provider):
                key_name = "GOOGLE_API_KEY" if provider == "gemini" else provider.upper() + "_API_KEY"
                model_name = {"zai": "glm-4.7-flash", "openrouter": "Vendor/Test:free"}.get(provider, "test-model")
                routes = self.configured({
                    "LLM_PROVIDER_ORDER": provider, key_name: "test-only-key",
                    provider.upper() + "_MODEL": model_name,
                    "CLOUDFLARE_ACCOUNT_ID": "synthetic-account",
                })
                self.assertEqual([label for label, _ in routes], [provider])
                self.assertEqual(routes[0][1].model_name, model_name)
                self.assertNotEqual(type(routes[0][1]).__name__, "MisconfiguredModel")

    def test_missing_or_placeholder_keys_are_skipped(self):
        routes = self.configured({
            "LLM_PROVIDER_ORDER": "mistral,groq,zai",
            "MISTRAL_API_KEY": "your_api_key_here", "GROQ_API_KEY": "paste_key_here",
            "ZAI_API_KEY": "replace_me",
        })
        self.assertEqual(routes, [])

    def test_blank_optional_settings_preserve_provider_defaults(self):
        routes = self.configured({
            "LLM_PROVIDER_ORDER": "groq", "GROQ_API_KEY": "test-key",
            "GROQ_MODEL": "", "GROQ_MODELS": "", "GROQ_BASE_URL": "",
            "GROQ_TOKEN_PARAMETER": "",
        })
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0][1].model_name, PROVIDERS["groq"].default_model)
        self.assertEqual(routes[0][1].base_url, PROVIDERS["groq"].base_url)

    def test_cloudflare_token_alias_is_supported(self):
        routes = self.configured({
            "LLM_PROVIDER_ORDER": "cloudflare",
            "CLOUDFLARE_API_TOKEN": "test-cf-token", "CLOUDFLARE_MODEL": "@cf/test-model",
            "CLOUDFLARE_ACCOUNT_ID": "synthetic-account",
        })
        self.assertEqual([label for label, _ in routes], ["cloudflare"])
        self.assertEqual(routes[0][1].api_key, "test-cf-token")
        self.assertEqual(routes[0][1].base_url,
                         "https://api.cloudflare.com/client/v4/accounts/synthetic-account/ai/v1")

    def test_primary_api_key_takes_precedence_over_token_alias(self):
        routes = self.configured({
            "LLM_PROVIDER_ORDER": "cloudflare", "CLOUDFLARE_MODEL": "@cf/test-model",
            "CLOUDFLARE_ACCOUNT_ID": "synthetic-account",
            "CLOUDFLARE_API_KEY": "test-primary-key", "CLOUDFLARE_API_TOKEN": "test-alias-token",
        })
        self.assertEqual(routes[0][1].api_key, "test-primary-key")

    def test_legacy_model_lists_keep_only_first_model_and_preserve_case(self):
        routes = self.configured({
            "LLM_PROVIDER_ORDER": "openrouter",
            "OPENROUTER_API_KEY": "test-only-key",
            "OPENROUTER_MODELS": "Vendor/Model-A:free, vendor/Model-B:free",
        })
        self.assertEqual([label for label, _ in routes], [
            "openrouter:Vendor/Model-A:free",
        ])
        self.assertEqual([model.model_name for _, model in routes], [
            "Vendor/Model-A:free",
        ])

    def test_explicit_model_route_overrides_model_environment(self):
        routes = self.configured({
            "LLM_PROVIDER_ORDER": "openrouter:Vendor/Chosen:free",
            "OPENROUTER_API_KEY": "test-only-key", "OPENROUTER_MODEL": "ignored-model",
            "OPENROUTER_MODELS": "ignored-a,ignored-b",
        })
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0][1].model_name, "Vendor/Chosen:free")

    def test_later_models_from_the_same_provider_are_skipped(self):
        routes = self.configured({
            "LLM_PROVIDER_ORDER": "groq,groq:shared-model",
            "GROQ_API_KEY": "test-only-key", "GROQ_MODELS": "shared-model,other-model",
        })
        self.assertEqual([model.model_name for _, model in routes], ["shared-model"])

    def test_same_model_at_different_providers_remains_two_routes(self):
        routes = self.configured({
            "LLM_PROVIDER_ORDER": "groq:shared-model,openrouter:shared-model",
            "GROQ_API_KEY": "test-groq-key", "OPENROUTER_API_KEY": "test-router-key",
        })
        self.assertEqual(len(routes), 2)
        self.assertNotEqual(routes[0][0], routes[1][0])

    def test_unknown_empty_and_duplicate_order_entries_are_rejected(self):
        for order in ["", "unknown", "groq,,gemini", ",groq", "groq,", "groq,groq", "groq:"]:
            with self.subTest(order=order):
                with self.assertRaises(ValueError):
                    self.configured({"LLM_PROVIDER_ORDER": order, "GROQ_API_KEY": "test-key"})

    def test_ollama_requires_an_explicit_model_but_no_api_key(self):
        self.assertEqual(self.configured({"LLM_PROVIDER_ORDER": "ollama"}), [])
        routes = self.configured({"LLM_PROVIDER_ORDER": "ollama", "OLLAMA_MODEL": "qwen3:8b"})
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0][1].model_name, "qwen3:8b")

    def test_ollama_model_list_chooses_one_model(self):
        routes = self.configured({
            "LLM_PROVIDER_ORDER": "ollama", "OLLAMA_MODELS": "qwen3:8b,gemma3:4b",
        })
        self.assertEqual([model.model_name for _, model in routes], ["qwen3:8b"])

    def test_custom_loopback_needs_model_and_url_but_not_key(self):
        routes = self.configured({
            "LLM_PROVIDER_ORDER": "custom", "CUSTOM_BASE_URL": "http://127.0.0.1:8080/v1",
            "CUSTOM_MODEL": "local-test-model",
        })
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0][1].model_name, "local-test-model")

    def test_custom_cloud_route_is_not_enabled_without_a_key(self):
        environment = {
            "LLM_PROVIDER_ORDER": "custom", "CUSTOM_BASE_URL": "https://example.invalid/v1",
            "CUSTOM_MODEL": "test-model",
        }
        try:
            routes = self.configured(environment)
        except ValueError:
            return
        if routes:
            # A deferred configuration failure is also safe when another route
            # is healthy, provided it cannot make an unauthenticated cloud call.
            with patch("llm_providers.httpx.Client") as client:
                for _, model in routes:
                    with self.assertRaises(ValueError):
                        model.invoke(MESSAGES)
                client.assert_not_called()

    def test_paid_only_providers_are_rejected_even_with_keys(self):
        for provider in ["openai", "anthropic", "deepseek", "cerebras", "together", "huggingface"]:
            with self.subTest(provider=provider), self.assertRaises(ValueError):
                self.configured({"LLM_PROVIDER_ORDER": provider, provider.upper() + "_API_KEY": "test-key"})

    def test_paid_models_and_remote_local_servers_never_make_requests(self):
        settings = [
            {"LLM_PROVIDER_ORDER": "zai:glm-5.3", "ZAI_API_KEY": "test-key"},
            {"LLM_PROVIDER_ORDER": "openrouter:vendor/paid-model", "OPENROUTER_API_KEY": "test-key"},
            {"LLM_PROVIDER_ORDER": "custom", "CUSTOM_API_KEY": "test-key",
             "CUSTOM_BASE_URL": "https://paid.example.invalid/v1", "CUSTOM_MODEL": "test"},
            {"LLM_PROVIDER_ORDER": "ollama", "OLLAMA_API_KEY": "test-key",
             "OLLAMA_BASE_URL": "https://ollama.com/v1", "OLLAMA_MODEL": "cloud-model"},
        ]
        for environment in settings:
            with self.subTest(environment=environment):
                model = self.configured(environment)[0][1]
                with patch("llm_providers.httpx.Client") as client, self.assertRaises(ValueError):
                    model.invoke(MESSAGES)
                client.assert_not_called()


class AdapterWireChecks(unittest.TestCase):
    def invoke_json(self, model, payload, *, timeout=35, status=200):
        captured = []

        def handler(request):
            captured.append(request)
            return httpx.Response(status, json=payload)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch("llm_providers.httpx.Client", return_value=client) as constructor:
            result = model.invoke(MESSAGES, timeout=timeout)
        self.assertEqual(len(captured), 1, "Each route must make one bounded attempt")
        return result, captured[0], constructor

    def chat_payload(self, content="Supported final answer", finish_reason="stop", **fields):
        return {"choices": [{"finish_reason": finish_reason, "message": {"content": content, **fields}}]}

    def test_generic_chat_maps_roles_and_uses_requested_timeout(self):
        result, request, constructor = self.invoke_json(
            HTTPChatModel("test-only-key", "test-model", "https://example.invalid/v1/"),
            self.chat_payload(reasoning="Private reasoning must not be displayed"), timeout=7,
        )
        self.assertEqual(str(request.url), "https://example.invalid/v1/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer test-only-key")
        body = json.loads(request.content)
        self.assertEqual(body["messages"], [
            {"role": "system", "content": MESSAGES[0][1]},
            {"role": "user", "content": MESSAGES[1][1]},
        ])
        self.assertEqual(body["model"], "test-model")
        self.assertFalse(body["stream"])
        self.assertNotIn("tools", body)
        self.assertEqual(constructor.call_args.kwargs["timeout"].read, 7)
        self.assertEqual(result.content, "Supported final answer")

    def test_zai_free_backup_reserves_its_budget_for_final_text(self):
        with patch.dict(os.environ, {"LLM_PROVIDER_ORDER": "zai", "ZAI_API_KEY": "test-zai"}, clear=True):
            model = configured_models()[0][1]
        _, request, _ = self.invoke_json(model, self.chat_payload())
        self.assertEqual(str(request.url), "https://api.z.ai/api/paas/v4/chat/completions")
        body = json.loads(request.content)
        self.assertEqual(body["model"], "glm-4.7-flash")
        self.assertEqual(body["thinking"], {"type": "disabled"})

    def test_generic_chat_honors_provider_token_and_body_options(self):
        _, request, _ = self.invoke_json(
            HTTPChatModel("test-key", "test-model", "https://example.invalid/v1",
                          token_parameter="max_completion_tokens",
                          extra_body={"reasoning_effort": "low", "include_reasoning": False}),
            self.chat_payload(),
        )
        body = json.loads(request.content)
        self.assertIn("max_completion_tokens", body)
        self.assertNotIn("max_tokens", body)
        self.assertFalse(body["include_reasoning"])

    def test_generic_chat_rejects_explicit_refusals(self):
        for payload in [self.chat_payload(refusal="Blocked"), self.chat_payload(finish_reason="content_filter"),
                        self.chat_payload(content=[{"type": "refusal", "refusal": "Blocked"}])]:
            with self.subTest(payload=payload):
                with self.assertRaises(ProviderRefusal):
                    self.invoke_json(HTTPChatModel("test-key", "test-model", "https://example.invalid/v1"), payload)

    def test_generic_chat_returns_only_final_content_blocks(self):
        payload = self.chat_payload(content=[
            {"type": "reasoning", "text": "Private chain of thought"},
            {"type": "text", "text": "Supported final answer"},
        ], reasoning_content="Also private")
        result, _, _ = self.invoke_json(
            HTTPChatModel("test-key", "test-model", "https://example.invalid/v1"), payload,
        )
        self.assertNotIn("Private", str(result.content))
        if isinstance(result.content, str):
            self.assertEqual(result.content, "Supported final answer")
        else:
            self.assertEqual(result.content, [{"type": "text", "text": "Supported final answer"}])

    def test_generic_chat_http_failure_is_not_an_answer(self):
        with self.assertRaises(httpx.HTTPStatusError):
            self.invoke_json(
                HTTPChatModel("test-key", "test-model", "https://example.invalid/v1"),
                {"error": {"message": "Synthetic private provider error"}}, status=429,
            )

    def test_unsafe_endpoint_urls_are_rejected(self):
        for url in [
            "http://example.invalid/v1", "https://user:password@example.invalid/v1",
            "https://example.invalid/v1?api_key=test", "https://example.invalid/v1#fragment",
            "file:///tmp/model", "ftp://example.invalid/v1", "http://127.0.0.1.evil.invalid/v1",
        ]:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    HTTPChatModel("test-key", "test-model", url)

    def test_loopback_http_endpoints_are_allowed(self):
        for url in ["http://127.0.0.1:11434/v1", "http://localhost:8080/v1", "http://[::1]:8080/v1"]:
            with self.subTest(url=url):
                model = HTTPChatModel("", "local-test-model", url)
                result, request, _ = self.invoke_json(model, self.chat_payload())
                self.assertEqual(result.content, "Supported final answer")
                self.assertTrue(str(request.url).endswith("/v1/chat/completions"))






    def test_cohere_native_v2_chat_keeps_grounding_and_ignores_reasoning(self):
        result, request, _ = self.invoke_json(CohereModel("test-key", "test-model"), {
            "finish_reason": "COMPLETE", "message": {"role": "assistant", "content": [
                {"type": "thinking", "text": "Private reasoning"},
                {"type": "text", "text": "Supported final answer"},
            ]},
        })
        self.assertEqual(str(request.url), "https://api.cohere.com/v2/chat")
        self.assertEqual(request.headers["Authorization"], "Bearer test-key")
        body = json.loads(request.content)
        self.assertEqual(body["messages"], [
            {"role": "system", "content": MESSAGES[0][1]},
            {"role": "user", "content": MESSAGES[1][1]},
        ])
        self.assertNotIn("tools", body)
        self.assertEqual(result.content, "Supported final answer")


if __name__ == "__main__":
    unittest.main(verbosity=2)
