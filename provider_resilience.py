"""Bound provider waits and recover from quota errors without changing chat state."""

import math
import re
import threading
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


_WORKER_SLOTS = threading.BoundedSemaphore(16)


class ProviderCapacityError(Exception):
    """This route already has enough calls running; try another route."""


class ProviderDeadlineError(TimeoutError):
    """The caller stopped waiting; a late result must never enter chat history."""


def invoke_before_deadline(callback, timeout, route_slots):
    """Run a stateless provider call with a bounded wait and bounded concurrency.

    SDK socket timeouts alone do not bound an entire call. Workers cannot mutate
    conversation state; only the calling request validates and commits a reply.
    Daemon workers can finish late, but their results are discarded after expiry.
    Per-route slots prevent a stalled primary from occupying all fallback slots.
    """
    if not route_slots.acquire(blocking=False):
        raise ProviderCapacityError()
    if not _WORKER_SLOTS.acquire(blocking=False):
        route_slots.release()
        raise ProviderCapacityError()
    result = Future()

    def run():
        try:
            result.set_result(callback())
        except Exception as error:
            result.set_exception(error)
        finally:
            route_slots.release()
            _WORKER_SLOTS.release()

    try:
        threading.Thread(target=run, name="kidney-chat-provider", daemon=True).start()
    except Exception:
        route_slots.release()
        _WORKER_SLOTS.release()
        raise
    try:
        return result.result(timeout=max(0, timeout))
    except FutureTimeoutError:
        raise ProviderDeadlineError() from None


def _retry_delay(value):
    if value is None:
        return None
    try:
        delay = float(str(value).removesuffix("s"))
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(str(value))
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            delay = (date - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return min(86400, max(1, delay)) if math.isfinite(delay) else None


def retry_after_seconds(error):
    """Respect Retry-After/Google RetryInfo; never expose raw provider details."""
    quota, delays, seen = False, [], set()
    current = error
    for _ in range(5):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        detail = str(current).lower()
        kind = type(current).__name__.lower()
        response = getattr(current, "response", None)
        code = getattr(current, "status_code", None) or getattr(current, "code", None)
        if callable(code):
            try:
                code = code()
            except Exception:
                code = None
        code = code or getattr(response, "status_code", None)
        quota |= str(code) in {"429", "RESOURCE_EXHAUSTED"} or any(
            token in detail or token in kind
            for token in ("429", "resource_exhausted", "resourceexhausted", "quota", "ratelimit")
        )
        headers = getattr(response, "headers", None)
        if headers is not None:
            delay = _retry_delay(headers.get("retry-after") or headers.get("Retry-After"))
            if delay is not None:
                delays.append(delay)
        # google.genai.errors.APIError stores the response JSON in .details.
        body = getattr(current, "details", None)
        if isinstance(body, dict):
            body = body.get("error", body)
            details = body.get("details", []) if isinstance(body, dict) else []
        elif isinstance(body, list):
            details = body
        else:
            details = []
        if not isinstance(details, (list, tuple)):
            details = []
        for item in details:
            if isinstance(item, dict) and "RetryInfo" in str(item.get("@type", "")):
                delay = _retry_delay(item.get("retryDelay"))
                if delay is not None:
                    delays.append(delay)
        # Some wrappers retain only Google's serialized RetryInfo message.
        retry = re.search(r"retry(?:delay| in)[\s\"':]*(\d+(?:\.\d+)?)\s*s", detail)
        if retry:
            delay = _retry_delay(retry.group(1))
            if delay is not None:
                delays.append(delay)
        current = current.__cause__ or current.__context__
    if delays:
        return max(delays)
    return 900 if quota else 60
