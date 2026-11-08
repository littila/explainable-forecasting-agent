import pathlib
import re
import time

import pytest

from solution import llm, pipeline, plan, priors, prompt, render, reply
from solution.corpus import load_unit
from solution.preflight import preflight
from solution.types import Unit

REPO = pathlib.Path(__file__).resolve().parents[2]


class _FakeClient:
    """Answers every call in the structured format: 3 reasons citing the first line of 8 words or more the prompt's
    documents show, then each entity quoting the first such line of its own documents. An entity in `bare` gets no claim on the first reply;
    a call asking for an entity in `hang` never returns; an entity in `cut` ends a first reply cut at the length limit
    and is left out of its correction; an entity in `drop` is left out of every call that asks for others too; an
    entity in `trunc` is left out of a call that asks for others too, which is then cut at the length limit."""

    url = "http://house.invalid/v1"
    model = "fake"
    route = llm.HOUSE
    unit: Unit
    hang: frozenset[str] = frozenset()
    bare: frozenset[str] = frozenset()
    cut: frozenset[str] = frozenset()
    drop: frozenset[str] = frozenset()
    trunc: frozenset[str] = frozenset()

    def __init__(self) -> None:
        self.trace: list[dict] = []
        self.requests = 0
        self.histories: list[list[dict]] = []

    def ask_text(self, system: str, user: str, *, history=None, deadline=None) -> llm.Text:
        self.requests += 1
        self.histories.append(list(history or []))
        if not (asked := re.search(r"in this order: (.+?)\. Reply", user)):
            return llm.Text("", "", "stop", 0)
        ids = asked.group(1).split(", ")
        if self.hang & set(ids):
            time.sleep(60)
        lines = [(doc, line.strip()) for doc, body in re.findall(r'<doc id="(D\d+)"[^>]*>\n(.*?)\n</doc>', user, re.DOTALL)
                 for line in body.splitlines() if len(line.split()) >= 8 and '"' not in line]
        labels = re.search(r"\nlabel: one of (.+?)\.\n", system)
        label = f"label: {labels.group(1).split(', ')[0]}\n" if labels else ""
        _, text = lines[0]
        reasons = [f'<reason n="{k}">\ncitation: "{text}"\npremise: p{k}.\nmechanism: m{k}.\n'
                   f"implication: i{k}.\n</reason>" for k in (1, 2, 3)]
        blocks = []
        for e in ids:
            if ((e in self.drop or e in self.trunc) and len(ids) > 1) or (e in self.cut and history):
                continue
            own = {f"D{k}" for k, d in enumerate(self.unit.docs, 1) if d.doc_id in render.owned(self.unit, e)}
            _, text = next(x for x in lines if x[0] in own)
            claim = "" if e in self.bare and not history else f'claim: "{text}"\n'
            blocks.append(f'<entity id="{e}">\n{claim}{label}point_forecast: 1.0\nint_lo: 0.0\nint_hi: 2.0\n</entity>')
        body = "<reasons>\n" + "\n".join(reasons) + "\n</reasons>\n<forecasts>\n" + "\n".join(blocks) + "\n</forecasts>"
        cut = self.cut & set(ids) and not history or self.trunc & set(ids) and len(ids) > 1
        return llm.Text(body, "", "length" if cut else "stop", 10)

    def spent(self) -> str:
        return "fake"


#: Pins figures measured in the model's own token counts: skipped when the tokenizer is not available
#: (`scripts/fetch_tokenizer.sh` fetches it; CI does).
needs_tokenizer = pytest.mark.skipif(render.counting() != "exact",
                                     reason="needs the model's tokenizer (scripts/fetch_tokenizer.sh)")


def _combined(unit: Unit, entity_id: str, model: float) -> float:
    """The point the answer gives an entity whose reply says `model`: its prior's when the prior decides it."""
    found = priors.priors(unit).get(entity_id)
    return found.point if found is not None and found.decides else model


def _fake(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, unit: Unit, **attrs) -> type[_FakeClient]:
    monkeypatch.setattr(llm, "Client", _FakeClient)
    monkeypatch.setattr(_FakeClient, "unit", unit, raising=False)
    for k, v in attrs.items():
        if k != "samples":
            monkeypatch.setattr(_FakeClient, k, v)
    monkeypatch.setattr(pipeline, "TRACE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "SAMPLES", (attrs.get("samples", 1),))
    return _FakeClient


def test_a_well_formed_run_answers_every_entity_with_a_citation_and_3_reasons(monkeypatch, tmp_path) -> None:
    unit = load_unit(REPO / "units/t4-cotpos-202411-us10")
    _fake(monkeypatch, tmp_path, unit)
    answer = pipeline.build_answer(unit)
    rows = answer["entity_predictions"]
    assert [r["entity_id"] for r in rows] == [e.entity_id for e in unit.entities]
    assert all(r["claims"] for r in rows)
    assert [r["reason_id"] for r in answer["submitted_reasons"]] == ["r1", "r2", "r3"]
    assert preflight(unit, answer) == []


def test_an_answer_call_gets_no_corrective_turn_and_a_missing_claim_is_repaired(monkeypatch, tmp_path) -> None:
    unit = load_unit(REPO / "units/t4-cpicomp-202410-us11")
    first = unit.entities[0].entity_id
    fake = _fake(monkeypatch, tmp_path, unit, bare=frozenset({first}))
    seen = []
    original = fake.ask_text

    def record(self, system, user, *, history=None, deadline=None):
        seen.append(history)
        return original(self, system, user, history=history, deadline=deadline)

    monkeypatch.setattr(fake, "ask_text", record)
    answer = pipeline.build_answer(unit)
    assert not any(seen)
    assert next(r for r in answer["entity_predictions"] if r["entity_id"] == first)["claims"]
    assert preflight(unit, answer) == []


def test_an_answer_that_never_returns_is_floored_at_the_deadline(monkeypatch, tmp_path) -> None:
    unit = load_unit(REPO / "units/t4-fomc-curve-20240918")
    _fake(monkeypatch, tmp_path, unit, hang=frozenset({unit.entities[-1].entity_id}))
    monkeypatch.setattr(pipeline, "MIN_CALL", 0.0)
    t0 = time.monotonic()
    answer = pipeline.build_answer(unit, deadline_at=t0 + 4.0, floor=True)
    assert time.monotonic() - t0 < 30
    assert all(r["claims"] for r in answer["entity_predictions"])
    assert "abandoned at the deadline" in answer["notes"]["failed_calls"].values()


def test_a_reasons_block_that_would_fail_the_schema_is_left_out(monkeypatch, tmp_path) -> None:
    unit = load_unit(REPO / "units/t4-fomc-curve-20240918")
    _fake(monkeypatch, tmp_path, unit)
    monkeypatch.setattr(reply, "selected", lambda *a: ([{"reason_id": "r1", "premise": "p"}], []))
    answer = pipeline.build_answer(unit)
    assert "submitted_reasons" not in answer
    assert answer["notes"]["reasons_left_out"]
    assert preflight(unit, answer) == []


def test_no_call_starts_with_less_than_the_minimum_left_and_the_floor_answers(monkeypatch, tmp_path) -> None:
    unit = load_unit(REPO / "units/t4-fomc-curve-20240918")
    _fake(monkeypatch, tmp_path, unit)
    answer = pipeline.build_answer(unit, deadline_at=time.monotonic() + 4.0, floor=True)
    assert all(v.startswith("not started") for v in answer["notes"]["failed_calls"].values())
    assert all(r["claims"] for r in answer["entity_predictions"])


def test_a_reply_cut_at_the_output_cap_keeps_its_whole_blocks(monkeypatch, tmp_path) -> None:
    unit = load_unit(REPO / "units/t4-cpicomp-202410-us11")
    first = unit.entities[0].entity_id
    fake = _fake(monkeypatch, tmp_path, unit, cut=frozenset({first}))
    seen = []
    original = fake.ask_text

    def record(self, system, user, *, history=None, deadline=None):
        seen.append(history)
        return original(self, system, user, history=history, deadline=deadline)

    monkeypatch.setattr(fake, "ask_text", record)
    answer = pipeline.build_answer(unit)
    row = next(r for r in answer["entity_predictions"] if r["entity_id"] == first)
    assert row["point_forecast"] == _combined(unit, first, 1.0) and row["claims"] and not any(seen)


def test_an_entity_no_reply_answers_takes_its_prior_without_another_call(monkeypatch, tmp_path) -> None:
    unit = load_unit(REPO / "units/t4-cpicomp-202410-us11")
    first = unit.entities[0].entity_id
    fake = _fake(monkeypatch, tmp_path, unit, drop=frozenset({first}))
    asked = []
    original = fake.ask_text

    def record(self, system, user, *, history=None, deadline=None):
        asked.append(user)
        return original(self, system, user, history=history, deadline=deadline)

    monkeypatch.setattr(fake, "ask_text", record)
    answer = pipeline.build_answer(unit)
    row = next(r for r in answer["entity_predictions"] if r["entity_id"] == first)
    assert len(asked) == len(plan.calls(unit))
    assert row["point_forecast"] == priors.priors(unit)[first].point and row["claims"]
    assert preflight(unit, answer) == []


def test_a_unit_whose_calls_all_fail_is_not_asked_again_and_gets_the_floor(monkeypatch, tmp_path) -> None:
    unit = load_unit(REPO / "units/t4-macrorev-20240930-us6")
    fake = _fake(monkeypatch, tmp_path, unit)
    made = []

    def refused(self, system, user, *, history=None, deadline=None):
        made.append(1)
        return llm.Text(None, "APIConnectionError: Connection error.")

    monkeypatch.setattr(fake, "ask_text", refused)
    answer = pipeline.build_answer(unit, floor=True)
    assert len(made) == len(plan.calls(unit))
    rows = {r["entity_id"]: r for r in answer["entity_predictions"]}
    found = priors.priors(unit)
    assert all(rows[e]["point_forecast"] == f.point for e, f in found.items())
    assert preflight(unit, answer) == []


def test_a_small_unit_is_answered_from_three_samples_merged(monkeypatch, tmp_path) -> None:
    unit = load_unit(REPO / "units/t4-cotpos-202411-us10")
    fake = _fake(monkeypatch, tmp_path, unit, samples=3)
    names = []
    original = fake.ask_text

    def record(self, system, user, *, history=None, deadline=None):
        names.append(user)
        return original(self, system, user, history=history, deadline=deadline)

    monkeypatch.setattr(fake, "ask_text", record)
    answer = pipeline.build_answer(unit)
    assert answer["notes"]["samples"] == 3 and len(names) == 3 * len(plan.calls(unit))
    assert preflight(unit, answer) == []


def test_samples_merge_to_the_median_sample_among_the_majority_label() -> None:
    a = {"E": {"claims": ["a"], "point_forecast": 1.0, "int_lo": 0.0, "int_hi": 2.0, "label": "up"}}
    b = {"E": {"claims": ["b"], "point_forecast": 5.0, "int_lo": 4.0, "int_hi": 6.0, "label": "down"}}
    c = {"E": {"claims": ["c"], "point_forecast": 2.0, "int_lo": 1.0, "int_hi": 4.0, "label": "down"}}
    out = reply.merge([a, b, c])["E"]
    assert (out["point_forecast"], out["int_lo"], out["int_hi"], out["label"], out["claims"]) == (2.0, 1.0, 6.0, "down", ["c"])


def test_the_merged_interval_spans_the_samples_and_holds_the_median_point() -> None:
    near = {"E": {"claims": ["n"], "point_forecast": 3.2, "int_lo": 2.8, "int_hi": 3.6}}
    far = {"E": {"claims": ["f"], "point_forecast": 5.3, "int_lo": 4.8, "int_hi": 5.8}}
    out = reply.merge([far, near, dict(near)])["E"]
    assert (out["point_forecast"], out["int_lo"], out["int_hi"], out["claims"]) == (3.2, 2.8, 5.8, ["n"])
    same = reply.merge([near, dict(near), dict(near)])["E"]
    assert (same["int_lo"], same["int_hi"]) == (2.8, 3.6)


def test_a_call_is_admitted_within_the_count_and_token_caps_and_a_larger_one_alone(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "IN_FLIGHT", 4)
    monkeypatch.setattr(pipeline, "TOKENS_IN_FLIGHT", 200_000)
    assert pipeline.admits([], 250_000) and pipeline.admits([90_000], 90_000)
    assert not pipeline.admits([90_000, 90_000], 90_000) and not pipeline.admits([10] * 4, 10)


def test_the_pool_starts_a_call_as_soon_as_one_ends(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "IN_FLIGHT", 2)
    monkeypatch.setattr(pipeline, "POLL", 0.01)
    monkeypatch.setattr(pipeline.render, "prompt_tokens", lambda text: 1)
    seconds = {"slow": 1.0, "a": 0.1, "b": 0.1, "c": 0.1}
    started, ended = {}, {}

    def send(client, call, texts, failures, corrected, end, check, correction, left):
        started[call.name] = time.monotonic()
        time.sleep(seconds[call.name])
        ended[call.name] = time.monotonic()
        texts[call.name] = ["ok"]

    monkeypatch.setattr(pipeline, "_send", send)
    calls = [plan.Call(n, [n], "", n) for n in seconds]
    texts, failures, _ = pipeline._run(_FakeClient(), calls, None, lambda *a: [], "")
    assert set(texts) == set(seconds) and failures == {}
    assert started["c"] < ended["slow"]


def test_the_most_odd_samples_that_fit_the_call_budget_are_drawn() -> None:
    assert [pipeline.samples_for(n) for n in (1, 2, 3, 4, 5, 6, 8)] == [5, 5, 5, 3, 3, 1, 1]



def test_a_unit_whose_points_all_come_from_priors_draws_one_sample(monkeypatch, tmp_path) -> None:
    unit = load_unit(REPO / "units/t4-macrorev-20240930-us6")
    _fake(monkeypatch, tmp_path, unit, samples=3)
    answer = pipeline.build_answer(unit)
    assert answer["notes"]["samples"] == 1 and preflight(unit, answer) == []


def test_a_forecast_code_decides_is_shown_on_its_entitys_line_before_the_model_writes(monkeypatch, tmp_path) -> None:
    unit = load_unit(REPO / "units/t4-macrorev-20240930-us6")
    fake = _fake(monkeypatch, tmp_path, unit)
    users = []
    original = fake.ask_text

    def record(self, system, user, *, history=None, deadline=None):
        users.append(user)
        return original(self, system, user, history=history, deadline=deadline)

    monkeypatch.setattr(fake, "ask_text", record)
    answer = pipeline.build_answer(unit)
    found = priors.priors(unit)
    shown = "\n".join(users)
    assert all(f"| decided forecast: {priors.decided(unit, f)}" in shown for f in found.values())
    assert prompt.DECIDED.strip() in shown and preflight(unit, answer) == []


def test_samples_shrink_to_the_requests_left() -> None:
    assert [pipeline.samples_for(2, room) for room in (25, 9, 6, 5)] == [5, 3, 3, 1]


def test_no_unit_takes_more_than_the_request_cap(monkeypatch, tmp_path) -> None:
    unit = load_unit(REPO / "units/t4-credit-event-2023")
    fake = _fake(monkeypatch, tmp_path, unit)
    made = []
    answer = fake.ask_text

    def counted(self, system, user, *, history=None, deadline=None):
        made.append(user)
        return answer(self, system, user, history=history, deadline=deadline)

    monkeypatch.setattr(fake, "ask_text", counted)
    monkeypatch.setattr(llm, "MAX_REQUESTS", 4)
    answer_out = pipeline.build_answer(unit, floor=True)
    assert len(made) <= 4
    assert sorted(e for u in made for e in re.search(r"in this order: (.+?)\. Reply", u).group(1).split(", ")) == \
        sorted(e.entity_id for e in unit.entities)
    assert preflight(unit, answer_out) == []


def test_copies_leave_the_answer_calls_their_share(monkeypatch, tmp_path) -> None:
    unit = load_unit(REPO / "units/t4-eps-growth-2024Q3-banks")
    fake = _fake(monkeypatch, tmp_path, unit)
    made = []
    answer = fake.ask_text

    def counted(self, system, user, *, history=None, deadline=None):
        made.append("copy" if system.startswith("You copy") else "answer")
        return answer(self, system, user, history=history, deadline=deadline)

    monkeypatch.setattr(fake, "ask_text", counted)
    pipeline.build_answer(unit, floor=True)
    assert made.count("copy") <= pipeline.COPY_REQUESTS
    assert made.count("answer") >= len(unit.entities) and len(made) <= llm.MAX_REQUESTS


@needs_tokenizer
def test_an_entity_with_no_reply_and_no_prior_is_asked_once_more(monkeypatch, tmp_path) -> None:
    unit = load_unit(REPO / "units/t4-postearn-20240201-megacap")
    first = unit.entities[0].entity_id
    fake = _fake(monkeypatch, tmp_path, unit, drop=frozenset({first}))
    asked = []
    original = fake.ask_text

    def record(self, system, user, *, history=None, deadline=None):
        asked.append(re.search(r"in this order: (.+?)\. Reply", user).group(1))
        return original(self, system, user, history=history, deadline=deadline)

    monkeypatch.setattr(fake, "ask_text", record)
    answer = pipeline.build_answer(unit, floor=True)
    assert first not in priors.priors(unit) and asked[-1] == first
    row = next(r for r in answer["entity_predictions"] if r["entity_id"] == first)
    assert row["point_forecast"] == 1.0 and preflight(unit, answer) == []


def test_a_trace_the_disk_refuses_does_not_cost_the_answer(monkeypatch, tmp_path) -> None:
    unit = load_unit(REPO / "units/t4-cotpos-202411-us10")
    _fake(monkeypatch, tmp_path, unit)
    blocked = tmp_path / "file"
    blocked.write_text("")
    monkeypatch.setattr(pipeline, "TRACE_DIR", blocked / "traces")
    answer = pipeline.build_answer(unit, floor=True)
    assert len(answer["entity_predictions"]) == len(unit.entities) and preflight(unit, answer) == []
