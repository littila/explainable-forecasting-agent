"""The model client: OpenAI-compatible chat completions.

`MODEL_ENDPOINT` selects the route. Unset is development on OpenRouter, the same model id the submission is scored on;
the House container is whatever the harness hands us, at `$MODEL_ENDPOINT/v1` with `$MODEL_TOKEN` (SUBMISSION_CLI.md).
Answers and copies are plain text (`ask_text`), read by `reply` and `extract`. `ask`, a reply constrained by a pydantic
model through `response_format`, is kept from earlier versions; the pipeline does not use it.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import threading
import time
from dataclasses import dataclass, field
from typing import NamedTuple
from urllib.parse import urlparse

from pydantic import BaseModel

REPO = pathlib.Path(__file__).resolve().parents[1]
OPENROUTER = "https://openrouter.ai/api/v1"

#: SUBMISSION_CLI.md: 25 admitted requests per unit, at most 4,000 output tokens per call.
MAX_REQUESTS, MAX_OUTPUT = 25, 4_000
#: The harness sets QFBENCH_SEED and rule 4 asks for the seed pinned. OpenRouter does not honour it.
SEED = int(os.environ.get("QFBENCH_SEED") or 20260731)
#: Greedy, for stable predictions (2026-09-28): at 0, EXAMPLE's label held on 6 of 6 OpenRouter runs.
TEMPERATURE, TOP_P = 0.0, 0.95
#: A call with no deadline (development) is bounded by this.
CALL_TIMEOUT = 900.0
#: Waits before re-sending after a transient failure: three 503s inside two seconds once lost a unit.
BACKOFF = (2.0, 8.0, 20.0)
#: On OpenRouter, where DekaLLM's shared pool answers most first attempts with 429 (m8m, 2026-09-29).
PATIENT = (2.0, 5.0, 10.0, 20.0, 30.0, 60.0)
#: No retry starts with less than this left before the deadline: an answer call takes 25-200 s.
RETRY_MIN_LEFT = 60.0
#: After this many calls in a row fail as unreachable (UNREACHABLE, each after its own retries), the House is taken as
#: down: later calls fail at once and the unit is answered from what is in hand. A refusal the House may lift (a rate
#: limit, an overload) never counts: it is retried with THROTTLED waits instead.
BREAK_AFTER = 3
UNREACHABLE = ("APIConnectionError", "APITimeoutError")
#: Waits before retrying a refusal the House may lift (a rate limit, an overload, a server error); a Retry-After
#: header is waited instead when it is longer.
THROTTLED = (5.0, 15.0, 30.0, 60.0)
#: Upstream statuses that clear on their own. 520/522/524 are OpenRouter's upstream failures.
TRANSIENT_STATUS = (429, 500, 502, 503, 504, 520, 522, 524)
#: A truncated reply this full of whitespace is a decoder loop, not a long answer (healthy: 20%,
#: loops: 69-99%).
WHITESPACE_LOOP = 0.5


@dataclass(frozen=True)
class Route:
    host: str
    token_env: str | None
    model: str
    #: The payload that turns thinking off: on the 120B it spends the whole 4,000-token cap.
    thinking_off: dict
    #: Provider routing sent with every request.
    routing: dict = field(default_factory=dict)
    backoff: tuple[float, ...] = BACKOFF


#: Every route plans against the House model's 262,144-token window (`render.WINDOW`); a local tag must hold it:
#: `nemotron-3-nano-256k:4b` is nemotron-3-nano:4b with num_ctx 262144 (8.3 GB, all GPU).
ROUTES = (
    Route("localhost", None, "nemotron-3-nano-256k:4b", {"reasoning_effort": "none"}),
    Route("127.0.0.1", None, "nemotron-3-nano-256k:4b", {"reasoning_effort": "none"}),
    # OpenRouter picks the provider, among those that enforce the schema: without this, a request can reach
    # one that does not (DeepInfra, 2026-09-29: its replies ignored the schema 3 times of 3)
    Route("openrouter.ai", "OPENROUTER_API_KEY", "nvidia/nemotron-3-super-120b-a12b:free",
          {"reasoning": {"effort": "none"}}, {"provider": {"require_parameters": True}}, PATIENT),
)
#: Any other host is the House container, whose address is not known in advance.
HOUSE = Route("", "MODEL_TOKEN", "house", {"chat_template_kwargs": {"enable_thinking": False}})


class Reply(NamedTuple):
    value: BaseModel | None
    why: str
    out_tokens: int = 0


class Text(NamedTuple):
    """A plain-text reply: its text (None on failure), why it failed, how it finished, its output tokens."""
    text: str | None
    why: str
    finish: str = ""
    out_tokens: int = 0


def _env(name: str) -> str | None:
    """The process env, else the untracked `.env` (development keys; excluded from the image)."""
    if os.environ.get(name):
        return os.environ[name]
    f = REPO / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == name:
                return v.strip().strip("\"'")
    return None


def schema(model: type[BaseModel]) -> dict:
    """The model's JSON schema as sent: references inlined, titles dropped, a one-value Literal as a
    one-item enum (the form every route has accepted)."""
    doc = model.model_json_schema()
    defs = doc.pop("$defs", {})

    def inline(node: object) -> object:
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(defs[node["$ref"].rsplit("/", 1)[-1]])
            if "const" in node:
                node = {**{k: v for k, v in node.items() if k != "const"}, "enum": [node["const"]]}
            return {k: inline(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node

    return inline(doc)


_STATUS = re.compile(rf"\b({'|'.join(str(c) for c in TRANSIENT_STATUS)})\b")


def _wait(exc: BaseException | None, base: float, deadline: float | None) -> float:
    """How long to wait before a retry: `base`, or the reply's Retry-After when longer, within what the clock leaves
    for the call after it."""
    response = getattr(exc, "response", None)
    try:
        asked = float(getattr(response, "headers", {}).get("retry-after", 0) or 0)
    except (TypeError, ValueError):
        asked = 0.0
    wait = max(base, asked)
    return wait if deadline is None else max(0.0, min(wait, deadline - time.monotonic() - RETRY_MIN_LEFT))


def _transient(text: str) -> bool:
    words = any(w in text.lower() for w in ("overloaded", "rate limit"))
    return (words or bool(_STATUS.search(text))) and "per-day" not in text


class Client:
    def __init__(self) -> None:
        from openai import OpenAI

        base = (_env("MODEL_ENDPOINT") or OPENROUTER).rstrip("/")
        if not urlparse(base).path:
            base += "/v1"                           # the House contract: an origin, served at /v1
        self.route = next((r for r in ROUTES if r.host in base), HOUSE)
        token = _env(self.route.token_env) if self.route.token_env else None
        if self.route.token_env and not token:
            raise RuntimeError(f"{self.route.token_env} is not set (endpoint {base})")
        self.url = base
        self.model = _env("MODEL_NAME") or self.route.model
        # the SDK's own retries would run past the clock and spend admitted requests uncounted
        self.client = OpenAI(base_url=base, api_key=token or "EMPTY", max_retries=0)
        self.requests = self.input_tokens = self.output_tokens = 0
        #: Every request and reply, verbatim, for the unit being answered.
        self.trace: list[dict] = []
        #: Calls in a row that failed; at BREAK_AFTER no further call is made.
        self.failed_in_a_row = 0
        self._lock = threading.Lock()

    def _may_retry(self, attempt: int, deadline: float | None, waits: tuple[float, ...] | None = None) -> bool:
        return (attempt < len(self.route.backoff if waits is None else waits) and self.requests < MAX_REQUESTS
                and (deadline is None or deadline - time.monotonic() > RETRY_MIN_LEFT))

    def ask(self, system: str, user: str, model: type[BaseModel], *, deadline: float | None = None) -> Reply:
        """One answer call, retried on transient failures. A reply that does not validate is resampled
        once (a provider without schema enforcement, a whitespace loop); one that ran out of room is not,
        since the same request overflows again."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        response_format = {"type": "json_schema", "json_schema": {"name": "unit_answer", "strict": True,
                                                                  "schema": schema(model)}}
        extra_body = {**self.route.thinking_off, **self.route.routing}
        resampled = False
        for attempt in range(max(len(self.route.backoff), len(THROTTLED)) + 1):
            timeout = CALL_TIMEOUT if deadline is None else max(5.0, deadline - time.monotonic())
            entry: dict = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "attempt": attempt, "url": self.url,
                           "model": self.model, "params": {"max_tokens": MAX_OUTPUT, "temperature": TEMPERATURE,
                                                           "top_p": TOP_P, "seed": SEED, "timeout": timeout,
                                                           "extra_body": extra_body},
                           "response_format": response_format, "messages": messages}
            with self._lock:
                self.trace.append(entry)
            t0 = time.time()
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=MAX_OUTPUT, temperature=TEMPERATURE,
                    top_p=TOP_P, seed=SEED, timeout=timeout, extra_body=extra_body,
                    response_format=response_format)
            except Exception as exc:
                if type(exc).__name__ == "OutOfTime":
                    raise
                text = str(exc)
                entry["secs"] = round(time.time() - t0, 1)
                entry["exception"] = {"type": type(exc).__name__, "message": text[:4000]}
                # refused before admission (a malformed request) costs no slot; anything sent does
                self._count(0 if isinstance(exc, (ValueError, TypeError)) else 1)
                # `APITimeoutError` is our own timeout, spent, and is deliberately not matched
                dropped = type(exc).__name__ == "APIConnectionError"
                waits = self.route.backoff if dropped else THROTTLED
                if (dropped or _transient(text)) and self._may_retry(attempt, deadline, waits):
                    time.sleep(_wait(exc, waits[attempt], deadline))
                    continue
                quota = "quota exhausted (not retryable): " if "per-day" in text else ""
                return Reply(None, f"{quota}{type(exc).__name__}: {text[:110]}")
            self._count(1)
            entry["secs"] = round(time.time() - t0, 1)
            entry["response"] = r.model_dump()
            out = (r.usage.completion_tokens or 0) if r.usage else 0
            if r.usage:
                with self._lock:
                    self.input_tokens += r.usage.prompt_tokens or 0
                    self.output_tokens += out
            # an upstream failure can arrive as a 200: `choices: null` with an error, or a choice
            # finished with "error" (m8c fomc lost three entities to a 502 inside the choice)
            error = str(getattr(r, "error", "") or "") if not r.choices else (
                str(getattr(r.choices[0], "error", "") or "") if r.choices[0].finish_reason == "error" else "")
            if error and _transient(error) and self._may_retry(attempt, deadline, THROTTLED):
                time.sleep(_wait(None, THROTTLED[attempt], deadline))
                continue
            if not r.choices:
                return Reply(None, f"upstream error: {error[:90]}", out)
            choice = r.choices[0]
            body = choice.message.content or ""
            if not body:
                return Reply(None, f"empty reply (finish={choice.finish_reason})", out)
            try:
                # a raw tab or newline copied from a filing into a string is not worth a call
                return Reply(model.model_validate(json.loads(body, strict=False)), "", out)
            except ValueError as exc:  # a JSON or validation error
                kind = "invalid"
                if choice.finish_reason == "length":
                    kind = "degenerate" if sum(c.isspace() for c in body) / len(body) >= WHITESPACE_LOOP else "length"
                if kind != "length" and not resampled and self._may_retry(attempt, deadline):
                    resampled = True
                    continue
                return Reply(None, f"{kind} (finish={choice.finish_reason}): {str(exc).splitlines()[0][:100]}", out)
        return Reply(None, "retries exhausted")

    def ask_text(self, system: str, user: str, *, history: list[dict] | None = None,
                 deadline: float | None = None) -> Text:
        """One plain-text call (no reply schema), retried on transient failures within the request cap and the
        clock; none once BREAK_AFTER calls in a row have failed. `history` continues the conversation after the user
        text (a reply and a corrective message)."""
        if self.failed_in_a_row >= BREAK_AFTER:
            return Text(None, f"House taken as down: {self.failed_in_a_row} calls in a row failed")
        reply = self._ask_text(system, user, history, deadline)
        with self._lock:
            if reply.text is not None:
                self.failed_in_a_row = 0
            elif reply.why.startswith(UNREACHABLE):
                self.failed_in_a_row += 1
        return reply

    def _ask_text(self, system: str, user: str, history: list[dict] | None, deadline: float | None) -> Text:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}, *(history or [])]
        extra_body = {**self.route.thinking_off, **self.route.routing}
        for attempt in range(max(len(self.route.backoff), len(THROTTLED)) + 1):
            timeout = CALL_TIMEOUT if deadline is None else max(5.0, deadline - time.monotonic())
            entry: dict = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "attempt": attempt, "url": self.url,
                           "model": self.model, "params": {"max_tokens": MAX_OUTPUT, "temperature": TEMPERATURE,
                                                           "top_p": TOP_P, "seed": SEED, "timeout": timeout,
                                                           "extra_body": extra_body},
                           "messages": messages}
            with self._lock:
                self.trace.append(entry)
            t0 = time.time()
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=MAX_OUTPUT, temperature=TEMPERATURE,
                    top_p=TOP_P, seed=SEED, timeout=timeout, extra_body=extra_body)
            except Exception as exc:
                if type(exc).__name__ == "OutOfTime":
                    raise
                text = str(exc)
                entry["secs"] = round(time.time() - t0, 1)
                entry["exception"] = {"type": type(exc).__name__, "message": text[:4000]}
                self._count(0 if isinstance(exc, (ValueError, TypeError)) else 1)
                dropped = type(exc).__name__ == "APIConnectionError"
                waits = self.route.backoff if dropped else THROTTLED
                if (dropped or _transient(text)) and self._may_retry(attempt, deadline, waits):
                    time.sleep(_wait(exc, waits[attempt], deadline))
                    continue
                quota = "quota exhausted (not retryable): " if "per-day" in text else ""
                return Text(None, f"{quota}{type(exc).__name__}: {text[:110]}")
            self._count(1)
            entry["secs"] = round(time.time() - t0, 1)
            entry["response"] = r.model_dump()
            out = (r.usage.completion_tokens or 0) if r.usage else 0
            if r.usage:
                with self._lock:
                    self.input_tokens += r.usage.prompt_tokens or 0
                    self.output_tokens += out
            error = str(getattr(r, "error", "") or "") if not r.choices else (
                str(getattr(r.choices[0], "error", "") or "") if r.choices[0].finish_reason == "error" else "")
            if error and _transient(error) and self._may_retry(attempt, deadline, THROTTLED):
                time.sleep(_wait(None, THROTTLED[attempt], deadline))
                continue
            if not r.choices:
                return Text(None, f"upstream error: {error[:90]}", "", out)
            choice = r.choices[0]
            body = choice.message.content or ""
            if not body:
                return Text(None, f"empty reply (finish={choice.finish_reason})", choice.finish_reason or "", out)
            return Text(body, "", choice.finish_reason or "", out)
        return Text(None, "retries exhausted")

    def _count(self, n: int) -> None:
        with self._lock:
            self.requests += n

    def spent(self) -> str:
        return (f"{self.requests} req, {self.input_tokens:,} in, {self.output_tokens:,} out "
                f"(caps: {MAX_REQUESTS} req, {MAX_OUTPUT:,} out/call)")
