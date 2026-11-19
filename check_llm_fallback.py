"""Offline provider failover checks; no real API keys or medical answers."""

import json
import os
import threading
import time
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx

from chatbot import ChatbotError, KidneyChatbot, SYSTEM_PROMPT
from llm_providers import GeminiModel, GroqModel, ProviderModel, ProviderRefusal, configured_models
from provider_resilience import retry_after_seconds


def reply(text="A supported test answer.", finish_reason="STOP"):
    return SimpleNamespace(content=text, response_metadata={"finish_reason": finish_reason})


class ProviderChecks(unittest.TestCase):
    def setUp(self):
        self.knowledge = SimpleNamespace(
            topics=[{"id": "GENERAL_001", "title": "Urgency", "text": "Conditional test reference."}],
            search=lambda *args, **kwargs: [],
        )
        self.primary = SimpleNamespace(invoke=Mock(return_value=reply()))
        self.backup = SimpleNamespace(invoke=Mock(return_value=reply("Fallback test answer.")))
        self.bot = KidneyChatbot(self.knowledge, providers=[
            ("gemini", self.primary), ("groq", self.backup),
        ])

    def test_success_does_not_send_data_to_fallback(self):
        self.assertEqual(self.bot.answer("Explain the result")["answer"], "A supported test answer.")
        self.backup.invoke.assert_not_called()

    def test_gemini_outage_does_not_attempt_a_second_gemini_model(self):
        second_gemini = SimpleNamespace(invoke=Mock(return_value=reply("Must not be attempted.")))
        self.primary.invoke.side_effect = RuntimeError("503 service unavailable")
        bot = KidneyChatbot(self.knowledge, providers=[
            ("gemini:primary", self.primary), ("Gemini:secondary", second_gemini), ("groq:backup", self.backup),
        ])
        self.assertEqual(bot.answer("Explain the result")["answer"], "Fallback test answer.")
        self.primary.invoke.assert_called_once()
        second_gemini.invoke.assert_not_called()
        self.backup.invoke.assert_called_once()

    def test_transient_and_configuration_failures_use_same_grounded_context(self):
        history = [{"role": "user", "content": "A previous question"},
                   {"role": "assistant", "content": "A previous answer"}]
        for error in [TimeoutError("private-key"), RuntimeError("429 quota private-key"),
                      RuntimeError("503 private-key"), RuntimeError("401 private-key"),
                      ImportError("private-key")]:
            with self.subTest(error=type(error).__name__):
                self.bot._retry_after.clear()
                self.primary.invoke.side_effect = error
                result = self.bot.answer("What next?", condition="stone", history=history)
                self.assertEqual(result["answer"], "Fallback test answer.")
                messages = self.backup.invoke.call_args.args[0]
                self.assertEqual(messages, self.primary.invoke.call_args.args[0])
                self.assertEqual(messages[0], ("system", SYSTEM_PROMPT))
                payload = json.loads(messages[1][1])
                self.assertEqual(payload["conversation_history"], history)
                self.assertEqual(payload["unconfirmed_result_label"], "stone")
                self.assertTrue(payload["knowledge_base_passages"])

    def test_cooldown_skips_failed_primary_then_allows_recovery(self):
        with patch("chatbot.time.monotonic", return_value=100):
            self.primary.invoke.side_effect = TimeoutError()
            self.bot.answer("First question")
            self.primary.invoke.side_effect = None
            self.bot.answer("Second question")
        self.assertEqual(self.primary.invoke.call_count, 1)
        self.assertEqual(self.backup.invoke.call_count, 2)
        with patch("chatbot.time.monotonic", return_value=161):
            self.bot.answer("Third question")
        self.assertEqual(self.primary.invoke.call_count, 2)
        self.assertEqual(self.backup.invoke.call_count, 2)

    def test_all_failures_are_sanitized_and_no_retry_storm(self):
        self.primary.invoke.side_effect = TimeoutError("private-key and private-question")
        self.backup.invoke.side_effect = RuntimeError("private-key and private-question")
        with self.assertLogs("chatbot", level="WARNING") as logs:
            for _ in range(2):
                with self.assertRaises(ChatbotError) as caught:
                    self.bot.answer("A kidney question")
                self.assertIn("busy right now", str(caught.exception))
                self.assertNotIn("private-", str(caught.exception))
        self.assertNotIn("private-", " ".join(logs.output))
        self.assertEqual(self.primary.invoke.call_count, 1)
        self.assertEqual(self.backup.invoke.call_count, 1)

    def test_incomplete_or_empty_answer_falls_back(self):
        for response in [reply(""), reply("Partial answer", "MAX_TOKENS"),
                         reply("Partial answer", "length"), reply("https://example.com"),
                         reply("<think>internal analysis only</think>")]:
            with self.subTest(response=response):
                self.bot._retry_after.clear()
                self.primary.invoke.return_value = response
                self.assertEqual(self.bot.answer("A question")["answer"], "Fallback test answer.")

    def test_explicit_safety_block_is_not_bypassed(self):
        self.primary.invoke.return_value = reply("", "SAFETY")
        with self.assertRaises(ChatbotError):
            self.bot.answer("A question")
        self.backup.invoke.assert_not_called()
        self.primary.invoke.side_effect = ProviderRefusal()
        with self.assertRaises(ChatbotError):
            self.bot.answer("A question")
        self.backup.invoke.assert_not_called()

    def test_reasoning_and_links_are_not_displayed(self):
        self.backup.invoke.return_value = reply(
            "<think>private reasoning</think>Useful text. [Read](https://example.com) [S001]"
        )
        self.primary.invoke.side_effect = TimeoutError()
        result = self.bot.answer("A question")
        self.assertEqual(result["answer"], "Useful text. Read")

    def test_invalid_question_never_calls_providers(self):
        with self.assertRaises(ValueError):
            self.bot.answer("")
        self.primary.invoke.assert_not_called()
        self.backup.invoke.assert_not_called()

    def test_groq_only_configuration_needs_no_google_key(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-only"}, clear=True):
            providers = configured_models()
        self.assertEqual([adapter.model_name for _, adapter in providers], [
            "openai/gpt-oss-120b",
        ])
        self.assertTrue(all(name.startswith("groq:") for name, _ in providers))

    def test_placeholders_skipped_and_order_can_be_reversed(self):
        with patch.dict(os.environ, {
            "GOOGLE_API_KEY": "your_key", "GROQ_API_KEY": "replace_me",
        }, clear=True):
            self.assertEqual(configured_models(), [])
        with patch.dict(os.environ, {
            "GOOGLE_API_KEY": "test-google", "GROQ_API_KEY": "test-groq",
            "LLM_PROVIDER_ORDER": "groq,gemini", "GROQ_MODEL": "custom-model",
        }, clear=True):
            providers = configured_models()
        self.assertEqual([name for name, _ in providers], ["groq", "gemini"])
        self.assertEqual(providers[0][1].model_name, "custom-model")

    def test_invalid_provider_order_rejected(self):
        for order in ["", "gemini,unknown", "groq,groq"]:
            with patch.dict(os.environ, {"LLM_PROVIDER_ORDER": order}, clear=True):
                with self.assertRaises(ValueError):
                    configured_models()

    def test_gemini_has_one_bounded_attempt(self):
        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as model:
            GeminiModel("test-only", "test-model").invoke([("human", "test")])
        self.assertEqual(model.call_args.kwargs["max_retries"], 1)
        self.assertEqual(model.call_args.kwargs["timeout"], 35)

    def test_groq_wire_format_excludes_reasoning(self):
        captured = []

        def handler(request):
            captured.append(request)
            return httpx.Response(200, json={"choices": [{
                "finish_reason": "stop", "message": {
                    "content": "Final answer", "reasoning": "private reasoning",
                },
            }]})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch("llm_providers.httpx.Client", return_value=client) as constructor:
            result = GroqModel("test-only", "openai/gpt-oss-120b").invoke([
                ("system", SYSTEM_PROMPT), ("human", "test payload"),
            ])
        request = captured[0]
        self.assertEqual(str(request.url), "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer test-only")
        body = json.loads(request.content)
        self.assertEqual(body["messages"][1], {"role": "user", "content": "test payload"})
        self.assertFalse(body["include_reasoning"])
        self.assertFalse(body["stream"])
        self.assertNotIn("tools", body)
        self.assertEqual(constructor.call_args.kwargs["timeout"].read, 35)
        self.assertEqual(result.content, "Final answer")

    def test_groq_http_failure_does_not_become_an_answer(self):
        client = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(429, json={"error": "private-key"})
        ))
        with patch("llm_providers.httpx.Client", return_value=client):
            with self.assertRaises(httpx.HTTPStatusError):
                GroqModel("test-only", "test-model").invoke([("human", "test")])

    def test_long_provider_chain_stops_when_request_budget_is_spent(self):
        clock = [0]

        class SlowProvider(ProviderModel):
            def __init__(self):
                self.timeouts = []

            def invoke(self, messages, timeout=35):
                self.timeouts.append(timeout)
                clock[0] = 101
                raise TimeoutError("private provider detail")

        slow = SlowProvider()
        bot = KidneyChatbot(self.knowledge, providers=[
            ("slow", slow), ("groq", self.backup),
        ])
        with patch("chatbot.time.monotonic", side_effect=lambda: clock[0]):
            with self.assertRaises(ChatbotError):
                bot.answer("A question")
        self.assertEqual(slow.timeouts, [35])
        self.backup.invoke.assert_not_called()

    def test_later_provider_timeout_uses_remaining_budget(self):
        clock = [0]

        class TimedProvider(ProviderModel):
            def __init__(self, fail=False):
                self.fail, self.timeouts = fail, []

            def invoke(self, messages, timeout=35):
                self.timeouts.append(timeout)
                if self.fail:
                    clock[0] = 85
                    raise TimeoutError()
                return reply("Fallback within budget.")

        first, last = TimedProvider(True), TimedProvider()
        bot = KidneyChatbot(self.knowledge, providers=[("primary", first), ("backup", last)])
        with patch("chatbot.time.monotonic", side_effect=lambda: clock[0]):
            self.assertEqual(bot.answer("A question")["answer"], "Fallback within budget.")
        self.assertEqual(first.timeouts, [35])
        self.assertEqual(last.timeouts, [15])

    def test_quota_retry_after_switches_immediately_and_skips_exhausted_primary(self):
        request = httpx.Request("POST", "https://provider.invalid/chat")
        response = httpx.Response(429, headers={"Retry-After": "120"}, request=request)
        self.primary.invoke.side_effect = httpx.HTTPStatusError("private-key", request=request, response=response)
        with patch("chatbot.time.monotonic", return_value=100):
            self.assertEqual(self.bot.answer("A question")["answer"], "Fallback test answer.")
        self.primary.invoke.side_effect = None
        with patch("chatbot.time.monotonic", return_value=219):
            self.bot.answer("Another question")
        self.assertEqual(self.primary.invoke.call_count, 1)
        with patch("chatbot.time.monotonic", return_value=221):
            self.assertEqual(self.bot.answer("A third question")["answer"], "A supported test answer.")
        self.assertEqual(self.primary.invoke.call_count, 2)

    def test_google_retry_info_survives_a_wrapped_provider_error(self):
        cause = RuntimeError("RESOURCE_EXHAUSTED private-detail")
        cause.code = 429
        cause.details = {"error": {"details": [{
            "@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "42s",
        }]}}
        wrapped = RuntimeError("Provider failed privately")
        wrapped.__cause__ = cause
        self.assertEqual(retry_after_seconds(wrapped), 42)
        self.assertEqual(retry_after_seconds(RuntimeError("429 quota")), 900)
        self.assertEqual(retry_after_seconds(TimeoutError()), 60)

    def test_retry_after_accepts_http_dates_and_ignores_invalid_hints(self):
        request = httpx.Request("POST", "https://provider.invalid/chat")
        response = httpx.Response(429, headers={"Retry-After": "Wed, 01 Jan 2025 00:02:00 GMT"}, request=request)
        error = httpx.HTTPStatusError("private-detail", request=request, response=response)
        with patch("provider_resilience.datetime") as clock:
            clock.now.return_value = datetime(2025, 1, 1, tzinfo=timezone.utc)
            self.assertEqual(retry_after_seconds(error), 120)
        for invalid in ("bad-delay", "NaN", "inf"):
            response.headers["Retry-After"] = invalid
            self.assertEqual(retry_after_seconds(error), 900)

    def test_hung_sdk_cannot_stop_fallback_and_late_answer_is_discarded(self):
        release, entered, finished = threading.Event(), threading.Event(), threading.Event()

        class IgnoringTimeout(ProviderModel):
            def invoke(self, messages, timeout=35):
                entered.set()
                release.wait(2)
                finished.set()
                return reply("Late primary answer must not be shown.")

        bot = KidneyChatbot(self.knowledge, providers=[("gemini", IgnoringTimeout()), ("groq", self.backup)])
        try:
            started = time.perf_counter()
            with patch("chatbot.PROVIDER_TIMEOUT", 0.03):
                result = bot.answer("A question")
            self.assertLess(time.perf_counter() - started, 1)
            self.assertTrue(entered.is_set())
            self.assertEqual(result["answer"], "Fallback test answer.")
            release.set()
            self.assertTrue(finished.wait(1))
            self.assertEqual(result["answer"], "Fallback test answer.")
        finally:
            release.set()

    def test_overall_deadline_returns_friendly_notice_when_sdk_never_finishes(self):
        release = threading.Event()

        def stall(messages):
            release.wait(2)
            return reply("Too late")

        self.primary.invoke.side_effect = stall
        self.backup.invoke.side_effect = stall
        try:
            started = time.perf_counter()
            with patch("chatbot.REQUEST_BUDGET_SECONDS", 0.03):
                with self.assertRaises(ChatbotError) as caught:
                    self.bot.answer("A question")
            self.assertLess(time.perf_counter() - started, 1)
            self.assertIn("busy right now", str(caught.exception))
            self.assertNotIn("timeout", str(caught.exception).lower())
            self.assertLessEqual(self.backup.invoke.call_count, 1)
        finally:
            release.set()

    def test_busy_primary_slots_do_not_block_healthy_backup(self):
        slots = self.bot._provider_slots["gemini"]
        slots.acquire()
        slots.acquire()
        try:
            self.assertEqual(self.bot.answer("A question")["answer"], "Fallback test answer.")
            self.primary.invoke.assert_not_called()
            self.assertNotIn("gemini", self.bot._retry_after)
        finally:
            slots.release()
            slots.release()

    def test_gemini_wait_is_shorter_when_another_provider_is_configured(self):
        timeouts = []

        class PromptProvider(ProviderModel):
            def invoke(self, messages, timeout=35):
                timeouts.append(timeout)
                return reply()

        bot = KidneyChatbot(self.knowledge, providers=[("gemini", PromptProvider()), ("groq", self.backup)])
        bot.answer("A question")
        self.assertEqual(timeouts, [15])

    def test_fifth_model_answers_after_four_limits_and_sixth_is_never_used(self):
        providers = []
        for index in range(6):
            invoke = Mock(side_effect=RuntimeError("429 quota")) if index < 4 else Mock(return_value=reply("Fifth model answer."))
            providers.append((f"test-{index}", SimpleNamespace(invoke=invoke)))
        bot = KidneyChatbot(self.knowledge, providers=providers)
        history = [{"role": "user", "content": "Previous test question"}]
        result = bot.answer("Follow-up test question", condition="stone", history=history)
        self.assertEqual(result["answer"], "Fifth model answer.")
        self.assertEqual(len(bot.providers), 5)
        expected_messages = providers[0][1].invoke.call_args.args[0]
        for _, provider in providers[:5]:
            provider.invoke.assert_called_once_with(expected_messages)
        providers[5][1].invoke.assert_not_called()
        self.assertEqual(json.loads(expected_messages[1][1])["conversation_history"], history)

    def test_slow_earlier_models_leave_time_for_the_fifth_model(self):
        clock, attempted = [0], []

        class SlowModel(ProviderModel):
            def __init__(self, index):
                self.index = index

            def invoke(self, messages, timeout=35):
                attempted.append((self.index, timeout))
                if self.index == 4:
                    return reply("Final backup within budget.")
                clock[0] += timeout
                raise TimeoutError()

        names = ["gemini:primary", "groq:large", "mistral:small", "zai:flash", "cloudflare:llama"]
        bot = KidneyChatbot(self.knowledge, providers=[(name, SlowModel(index)) for index, name in enumerate(names)])
        with patch("chatbot.time.monotonic", side_effect=lambda: clock[0]):
            result = bot.answer("A question")
        self.assertEqual(result["answer"], "Final backup within budget.")
        self.assertEqual([index for index, _ in attempted], [0, 1, 2, 3, 4])
        self.assertGreater(attempted[-1][1], 0)
        self.assertLessEqual(clock[0] + attempted[-1][1], 100)

    def test_older_inflight_results_cannot_remove_or_shorten_a_newer_quota_cooldown(self):
        for older_fails in (False, True):
            with self.subTest(older_request_fails=older_fails):
                entered, release = threading.Event(), threading.Event()
                count_lock = threading.Lock()
                calls, results, errors = [0], [], []
                request = httpx.Request("POST", "https://provider.invalid/chat")
                response = httpx.Response(429, headers={"Retry-After": "120"}, request=request)

                def invoke(messages):
                    with count_lock:
                        calls[0] += 1
                        call = calls[0]
                    if call == 1:
                        entered.set()
                        release.wait(5)
                        if older_fails:
                            raise TimeoutError()
                        return reply("Earlier primary answer.")
                    raise httpx.HTTPStatusError("Synthetic quota", request=request, response=response)

                primary = SimpleNamespace(invoke=Mock(side_effect=invoke))
                backup = SimpleNamespace(invoke=Mock(return_value=reply("Backup answer.")))
                bot = KidneyChatbot(self.knowledge, providers=[("gemini", primary), ("groq", backup)])

                def earlier_request():
                    try:
                        results.append(bot.answer("Earlier question"))
                    except Exception as error:
                        errors.append(error)

                worker = threading.Thread(target=earlier_request, daemon=True)
                try:
                    with patch("chatbot.time.monotonic", return_value=100):
                        worker.start()
                        self.assertTrue(entered.wait(2))
                        self.assertEqual(bot.answer("Later question")["answer"], "Backup answer.")
                        cooldown = bot._retry_after["gemini"]
                        release.set()
                        worker.join(2)
                        self.assertFalse(worker.is_alive())
                        self.assertFalse(errors)
                        self.assertEqual(bot._retry_after["gemini"], cooldown)
                        self.assertEqual(bot.answer("Next question")["answer"], "Backup answer.")
                        self.assertEqual(calls[0], 2)
                        self.assertEqual(results[0]["answer"], "Backup answer." if older_fails else "Earlier primary answer.")
                finally:
                    release.set()


if __name__ == "__main__":
    unittest.main(verbosity=2)
