import dataclasses
import time
from types import SimpleNamespace
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from solution import llm


class _Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: int
    kind: Literal["x"]


@pytest.fixture
def client(monkeypatch) -> llm.Client:
    monkeypatch.setenv("MODEL_ENDPOINT", "http://model:8443")
    monkeypatch.setenv("MODEL_TOKEN", "t")
    c = llm.Client()
    c.route = dataclasses.replace(c.route, backoff=(0.0, 0.0, 0.0))
    monkeypatch.setattr(llm, "THROTTLED", (0.0, 0.0, 0.0, 0.0))
    return c


def _choice(content: str, finish: str = "stop", error: object = None) -> SimpleNamespace:
    return SimpleNamespace(finish_reason=finish, error=error, message=SimpleNamespace(content=content))


def _reply(*choices: SimpleNamespace, error: object = None) -> SimpleNamespace:
    return SimpleNamespace(choices=list(choices) or None, error=error, model_dump=dict,
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def test_the_endpoint_selects_the_route(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_ENDPOINT", "http://model:8443")
    monkeypatch.setenv("MODEL_TOKEN", "t")
    house = llm.Client()
    assert house.url == "http://model:8443/v1" and house.route is llm.HOUSE      # origin + /v1
    monkeypatch.setenv("MODEL_ENDPOINT", "http://localhost:11434")
    assert llm.Client().route.host == "localhost"
    monkeypatch.setenv("MODEL_ENDPOINT", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert llm.Client().route.host == "openrouter.ai"
    assert llm.Client().url == "https://openrouter.ai/api/v1"                    # a path is kept as given


def test_a_missing_token_refuses_to_start(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_ENDPOINT", "http://model:8443")
    monkeypatch.delenv("MODEL_TOKEN", raising=False)
    monkeypatch.setattr(llm, "_env", lambda name: None if name == "MODEL_TOKEN" else "http://model:8443")
    with pytest.raises(RuntimeError, match="MODEL_TOKEN"):
        llm.Client()


def test_transient_upstream_statuses_are_retried_and_client_errors_are_not() -> None:
    for text in ("Error code: 520 - Provider returned error", "Error code: 504", "429 Too Many Requests",
                 "Service temporarily overloaded"):
        assert llm._transient(text), text
    for text in ("400 Bad Request", "invalid schema", "401 Unauthorized", "429 per-day quota exceeded",
                 "max_tokens 1500 exceeds the limit", "request id 5202 rejected"):
        assert not llm._transient(text), text


def test_a_retried_attempt_is_counted_and_the_cap_stops_retries(client) -> None:
    def limited(**kw):
        raise type("RateLimitError", (Exception,), {})("Error code: 429 - Provider returned error")

    client.client.chat.completions.create = limited
    assert client.ask("s", "u", _Answer).value is None
    assert client.requests == 1 + len(llm.THROTTLED)                # a rate limit takes the longer list, all charged
    client.requests = llm.MAX_REQUESTS - 1
    client.ask("s", "u", _Answer)
    assert client.requests == llm.MAX_REQUESTS                      # no retry past the cap


def test_a_dropped_connection_is_retried_and_our_own_timeout_is_not(client) -> None:
    connection = type("APIConnectionError", (Exception,), {})
    timeout = type("APITimeoutError", (connection,), {})
    for exc, calls in ((connection("Connection error."), 4), (timeout("Request timed out."), 1)):
        client.requests = 0
        client.client.chat.completions.create = lambda exc=exc, **kw: (_ for _ in ()).throw(exc)
        client.ask("s", "u", _Answer)
        assert client.requests == calls


def test_no_retry_starts_close_to_the_deadline(client) -> None:
    def failing(**kw):
        raise RuntimeError("Error code: 503")

    client.client.chat.completions.create = failing
    client.ask("s", "u", _Answer, deadline=time.monotonic() + llm.RETRY_MIN_LEFT - 1)
    assert client.requests == 1


def test_a_provider_error_inside_a_reply_is_retried(client) -> None:
    replies = iter([_reply(_choice('{"a": 1', "error", {"code": 502, "message": "Network connection lost."})),
                    _reply(_choice('{"a": 1, "kind": "x"}'))])
    client.client.chat.completions.create = lambda **kw: next(replies)
    reply = client.ask("s", "u", _Answer)
    assert (reply.value.a, reply.why, client.requests) == (1, "", 2)


def test_a_null_choices_error_is_retried(client) -> None:
    replies = iter([_reply(error={"message": "Service temporarily overloaded", "code": 503}),
                    _reply(_choice('{"a": 2, "kind": "x"}'))])
    client.client.chat.completions.create = lambda **kw: next(replies)
    assert client.ask("s", "u", _Answer).value.a == 2


def test_an_invalid_reply_is_resampled_once(client) -> None:
    client.client.chat.completions.create = lambda **kw: _reply(_choice('{"a": 1, "kind": "y"}'))
    reply = client.ask("s", "u", _Answer)
    assert reply.value is None and reply.why.startswith("invalid") and client.requests == 2
    replies = iter([_reply(_choice('{"a": 1,')), _reply(_choice('{"a": 5, "kind": "x"}'))])
    client.requests = 0
    client.client.chat.completions.create = lambda **kw: next(replies)
    assert client.ask("s", "u", _Answer).value.a == 5 and client.requests == 2


def test_a_reply_that_ran_out_of_room_is_not_resampled(client) -> None:
    client.client.chat.completions.create = lambda **kw: _reply(_choice('{"a": 1, "note": "' + "x" * 500, "length"))
    reply = client.ask("s", "u", _Answer)
    assert reply.why.startswith("length") and client.requests == 1


def test_a_truncated_whitespace_loop_is_named_degenerate(client) -> None:
    client.client.chat.completions.create = lambda **kw: _reply(_choice('{"a": 1,' + " " * 500, "length"))
    assert client.ask("s", "u", _Answer).why.startswith("degenerate")


def test_a_raw_control_character_in_a_string_is_tolerated(client) -> None:
    class Noted(BaseModel):
        note: str

    client.client.chat.completions.create = lambda **kw: _reply(_choice('{"note": "copied\tfrom a filing"}'))
    assert client.ask("s", "u", Noted).value.note == "copied\tfrom a filing"


def test_the_request_carries_the_schema_messages_and_sampling(client) -> None:
    sent = {}

    def capture(**kw):
        sent.update(kw)
        return _reply(_choice('{"a": 1, "kind": "x"}'))

    client.client.chat.completions.create = capture
    client.ask("sys", "usr", _Answer)
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["response_format"]["json_schema"]["schema"] == llm.schema(_Answer)
    assert (sent["temperature"], sent["seed"], sent["max_tokens"]) == (llm.TEMPERATURE, llm.SEED, llm.MAX_OUTPUT)
    assert sent["extra_body"] == llm.HOUSE.thinking_off                     # House: no routing
    assert len(client.trace) == 1 and client.trace[0]["response_format"] == sent["response_format"]


def test_the_sent_schema_is_flat_and_untitled() -> None:
    class Inner(BaseModel):
        x: Literal["only"]

    class Outer(BaseModel):
        items: list[Inner]

    doc = llm.schema(Outer)
    assert "$defs" not in doc and "title" not in str(doc)
    assert doc["properties"]["items"]["items"]["properties"]["x"] == {"type": "string", "enum": ["only"]}


def test_openrouter_routes_only_to_schema_enforcing_providers_and_waits_longer(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_ENDPOINT", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    client = llm.Client()
    sent = {}

    def capture(**kw):
        sent.update(kw)
        return _reply(_choice('{"a": 1, "kind": "x"}'))

    client.client.chat.completions.create = capture
    client.ask("s", "u", _Answer)
    assert sent["extra_body"]["reasoning"] == {"effort": "none"}
    assert sent["extra_body"]["provider"] == {"require_parameters": True}
    assert len(client.route.backoff) > len(llm.HOUSE.backoff)


def test_a_plain_text_call_sends_no_schema_and_carries_the_history(client) -> None:
    sent = {}

    def capture(**kw):
        sent.update(kw)
        return _reply(_choice("target: x\nA — a\nA, 1, 0, 2"))

    client.client.chat.completions.create = capture
    history = [{"role": "assistant", "content": "first"}, {"role": "user", "content": "fix it"}]
    got = client.ask_text("sys", "usr", history=history)
    assert got.text.startswith("target:") and got.finish == "stop" and got.out_tokens == 5
    assert [m["role"] for m in sent["messages"]] == ["system", "user", "assistant", "user"]
    assert "response_format" not in sent
    assert (sent["temperature"], sent["seed"], sent["max_tokens"]) == (llm.TEMPERATURE, llm.SEED, llm.MAX_OUTPUT)


def test_a_plain_text_call_retries_a_transient_failure_and_reports_the_finish(client) -> None:
    calls = []

    def flaky(**kw):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("Error code: 429 - rate limit")
        return _reply(_choice("cut short", finish="length"))

    client.client.chat.completions.create = flaky
    got = client.ask_text("sys", "usr")
    assert len(calls) == 2 and client.requests == 2
    assert (got.text, got.finish) == ("cut short", "length")


def test_an_empty_plain_text_reply_is_a_failure(client) -> None:
    client.client.chat.completions.create = lambda **kw: _reply(_choice(""))
    got = client.ask_text("sys", "usr")
    assert got.text is None and "empty reply" in got.why


def test_after_three_failed_calls_in_a_row_no_further_call_is_made(client) -> None:
    class APIConnectionError(Exception):
        pass

    made = []

    def refused(**kw):
        made.append(1)
        raise APIConnectionError("connection refused")

    client.client.chat.completions.create = refused
    for _ in range(llm.BREAK_AFTER):
        assert client.ask_text("s", "u").text is None
    before = len(made)
    reply = client.ask_text("s", "u")
    assert reply.text is None and "taken as down" in reply.why and len(made) == before


def test_a_rate_limit_never_takes_the_house_as_down(client) -> None:
    made = []

    def limited(**kw):
        made.append(1)
        raise type("RateLimitError", (Exception,), {})("Error code: 429 - rate limit")

    client.client.chat.completions.create = limited
    for _ in range(llm.BREAK_AFTER + 1):
        assert client.ask_text("s", "u").text is None
    assert client.failed_in_a_row == 0 and len(made) == (llm.BREAK_AFTER + 1) * (1 + len(llm.THROTTLED))


def test_a_retry_waits_for_retry_after_within_the_clock() -> None:
    import types
    exc = types.SimpleNamespace(response=types.SimpleNamespace(headers={"retry-after": "12"}))
    assert llm._wait(exc, 5.0, None) == 12.0
    assert llm._wait(None, 5.0, None) == 5.0
    assert llm._wait(exc, 5.0, time.monotonic() + llm.RETRY_MIN_LEFT + 3) <= 3.0
