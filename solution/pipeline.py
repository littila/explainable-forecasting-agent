"""One unit end to end: work out what the data decides, copy figures, ask, combine, cite, choose reasons.

Code first computes each entity's prior from the corpus's dated tables (`priors`), copies the latest level of a growth
target's quantity where a row gives its base (`extract`; a copy that fails its checks is sent back once), and reads
the filings' credit signals (`signals`); a forecast code decides is shown on its entity's line (`prompt`). The answer
calls (`plan`) are drawn 5, 3 or 1 times, at most IN_FLIGHT at once, with no corrective turn: an entity no reply
answers takes its prior, and one with no prior is asked once more. A yield-change unit then copies earlier levels for
its interval floor (`moves`). The replies are merged per entity (`reply.merge`), joined to the priors
(`priors.combine`), cited by quote in the original text (`cite`), assembled into rows no gate refuses (`assemble`),
and up to 3 reasons the grader would judge are chosen (`reply.selected`).

The budget: copies take at most COPY_REQUESTS of the 25 requests, the earlier-level copy keeps EARLIER_REQUESTS, the
answer calls take the rest, and the planner fits documents smaller rather than plan more calls than that. A unit no
plan fits is asked with its entity rows alone (`plan.rows_only`), and one whose copies cannot be planned goes without.

The clock: with a deadline, a call's timeout is what is left, none starts with less than MIN_CALL left, those in flight
at the deadline are abandoned, and reasons are chosen until REASONS_AFTER past it. With the floor on (submission) an
entity left without a citation cites what the roster cited most within its scope, else its own task row. The CLI sets
the deadline and the floor; the dev entry has neither.

    MODEL_NAME=... uv run python -m solution.pipeline units/<unit> [--tag=t]
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from functools import partial

from solution import (
    cite,
    extract,
    llm,
    moves,
    plan,
    priors,
    render,
    reply,
    signals,
)
from solution.assemble import assemble
from solution.corpus import load_unit
from solution.preflight import preflight
from solution.types import Unit

REPO = pathlib.Path(__file__).resolve().parents[1]
#: A reply's problems for the entities its call asks for: (text, finish reason, entity ids) -> problems.
Check = Callable[[str, str, list[str]], list[str]]
#: Calls sent at once, and the most prompt tokens they may carry together: a unit of large filings sends fewer at a
#: time, so a token-rate limit on the House is not met all at once.
IN_FLIGHT = 4
TOKENS_IN_FLIGHT = 200_000
#: How often the call pool looks for a finished call, in seconds.
POLL = 0.25
#: Samples of a unit's answer calls, combined per entity (`reply.merge`): the most of SAMPLES that fit in SAMPLE_CALLS
#: calls, which leaves the rest of the 25 requests for extraction; odd, so a label vote has no tie.
SAMPLES = (5, 3)
SAMPLE_CALLS = 15
#: Copy calls planned at most, and requests the copies may take with their corrections: at least half of the 25
#: requests are left for the answer calls.
COPY_CALLS = 8
COPY_REQUESTS = 12
#: Requests kept for the earlier-level copy that follows the answer calls (a call and its correction).
EARLIER_REQUESTS = 2
#: No call starts with less than this left before the deadline.
MIN_CALL = 30.0
#: Reasons are chosen until this long after the deadline at most; the CLI's alarm fires 30 s after it.
REASONS_AFTER = 10.0
TRACE_DIR = pathlib.Path(os.environ.get("T4_TRACE_DIR") or (REPO / ".local/traces"))
_TAKING = threading.Lock()


def _take(left: list[int]) -> bool:
    """One request from a run's budget `left` (a one-item list the run's threads share), when one is left."""
    with _TAKING:
        if left[0] <= 0:
            return False
        left[0] -= 1
        return True


def _send(client: llm.Client, call: plan.Call, texts: dict[str, list[str]], failures: dict[str, str],
          corrected: set[str], end: float | None, check: Check | None, correction: str, left: list[int]) -> None:
    t0 = time.time()
    first = client.ask_text(call.system, call.user, deadline=end)
    if first.text is None:
        failures[call.name] = first.why
        print(f"  call {call.name or '1'} {','.join(call.ids)}: FAILED in {time.time() - t0:.0f}s: {first.why}", flush=True)
        return
    texts[call.name] = [first.text]
    issues = check(first.text, first.finish, call.ids) if check else []
    remaining = None if end is None else end - time.monotonic()
    # a reply cut at the output cap comes back cut again when asked to answer in full
    cut = first.finish == "length"
    if issues and not cut and (remaining is None or remaining > MIN_CALL) and _take(left):
        second = client.ask_text(call.system, call.user, deadline=end,
                                 history=[{"role": "assistant", "content": first.text},
                                          {"role": "user", "content": correction.format(issues="; ".join(issues))}])
        if second.text is not None:
            texts[call.name] = [second.text, first.text]
            corrected.add(call.name)
    print(f"  call {call.name or '1'} {','.join(call.ids)}: ok in {time.time() - t0:.0f}s"
          f"{' (cut at the output cap)' if cut else ' (corrected: ' + '; '.join(issues) + ')' if issues else ''}",
          flush=True)


def _run(client: llm.Client, calls: list[plan.Call], end: float | None, check: Check | None = None,
         correction: str = "", limit: int = llm.MAX_REQUESTS) -> tuple[dict[str, list[str]], dict[str, str], set[str]]:
    """The calls in order, each started as soon as it is admitted (`admits`) beside those in flight; with a `check` (the
    problems of a reply for its call's entities) a reply that falls short is sent back once with `correction` while the
    client has made fewer than `limit` requests. None starts with less than MIN_CALL left, or once the calls started
    here would take the client to `limit`; those still in flight at the deadline are abandoned."""
    texts: dict[str, list[str]] = {}
    failures: dict[str, str] = {}
    corrected: set[str] = set()
    sizes = {c.name: render.prompt_tokens(c.system + c.user) for c in calls}
    pending, running = list(calls), {}
    budget = [limit - client.requests]
    while pending or running:
        while pending and admits([sizes[n] for n in running], sizes[pending[0].name]):
            call = pending.pop(0)
            left = None if end is None else end - time.monotonic()
            if left is not None and left < MIN_CALL:
                failures[call.name] = f"not started: {max(0.0, left):.0f}s left"
                print(f"  call {call.name or '1'}: NOT STARTED, {max(0.0, left):.0f}s left", flush=True)
                continue
            if not _take(budget):
                failures[call.name] = f"not started: the {limit}-request cap"
                print(f"  call {call.name or '1'}: NOT STARTED, the {limit}-request cap", flush=True)
                continue
            running[call.name] = threading.Thread(target=_send, daemon=True, args=(
                client, call, texts, failures, corrected, end, check, correction, budget))
            running[call.name].start()
        if end is not None and time.monotonic() >= end:
            for name in running:
                failures[name] = "abandoned at the deadline"
                print(f"  call {name or '1'}: ABANDONED at the deadline", flush=True)
            for call in pending:
                failures[call.name] = "not started: 0s left"
            break
        time.sleep(POLL)
        running = {n: t for n, t in running.items() if t.is_alive()}
    return dict(texts), dict(failures), set(corrected)


def admits(in_flight: list[int], size: int) -> bool:
    """Whether a call of `size` prompt tokens may start beside calls of `in_flight` tokens: at most IN_FLIGHT calls and
    TOKENS_IN_FLIGHT tokens at once, and a single larger call goes alone."""
    return not in_flight or (len(in_flight) < IN_FLIGHT and sum(in_flight) + size <= TOKENS_IN_FLIGHT)


def floor_answer(unit: Unit, why: str) -> dict:
    """The answer with no model reply: each entity's computed prior where it has one, else the floor point."""
    try:
        found = priors.priors(unit)
    except Exception as exc:  # noqa: BLE001 -- the floor must not fail on a prior
        print(f"  PRIORS failed: {exc}", flush=True)
        found = {}
    answer, done = answer_from(unit, [], floor=True, found=found)
    answer["notes"] = {"agent": "solution.pipeline floor", "failed": why, "priors": {e: f.method for e, f in found.items()},
                       "repairs": done["repairs"]}
    return answer


def cited(unit: Unit, preds: dict[str, dict]) -> tuple[dict[str, dict], list[str]]:
    """The answer rows from the parsed reply: each claim placed in the documents (`cite.place`) in its passage's own
    words, then the rows each prior used; the quotes found nowhere."""
    text = {d.doc_id: d.text for d in unit.docs}
    rows: dict[str, dict] = {}
    missing: list[str] = []
    for e, p in preds.items():
        placed = []
        for c in p["claims"]:
            if (got := cite.place(unit, e, c)) is None:
                missing.append(f"{e}: quote not found: {c.get('quote', c.get('node_id', ''))[:80]}")
            else:
                placed += got
        placed += [s | {"claim": cite.verbatim(text[s["doc_id"]], s["span_start"], s["span_end"])}
                   for s in p.get("spans", [])]
        rows[e] = {"point_forecast": p["point_forecast"], "interval_lo": p["int_lo"], "interval_hi": p["int_hi"],
                   "claims": placed, "label": p.get("label"), "reasoning": " ".join(p.get("reasoning", []))}
    return rows, missing


def _copies(client: llm.Client, unit: Unit, qualified: dict[str, tuple[str, float]], kind: extract.Kind,
            end: float | None) -> tuple[dict[str, extract.Level], list[str]]:
    """Levels of the `qualified` entities copied from their documents (`extract`), and what was left out."""
    if not qualified:
        return {}, []
    try:
        calls = extract.calls(unit, qualified, kind, most_calls=None if kind.dated else COPY_CALLS)
    except Exception as exc:  # noqa: BLE001 -- a unit is answered without copies rather than not at all
        return {}, [f"no extraction plan: {type(exc).__name__}: {exc}"]
    limit = llm.MAX_REQUESTS if kind.dated else COPY_REQUESTS
    if client.requests + len(calls) > limit:
        return {}, [f"{len(calls)} extraction calls would pass the request cap"]
    texts, failures, _ = _run(client, calls, end, partial(extract.problems, unit, qualified, kind=kind), kind.correction,
                              limit)
    got, left_out = {}, [f"{name}: {why}" for name, why in failures.items()]
    for c in calls:
        for text in texts.get(c.name, []):
            levels, dropped = extract.levels(unit, text, {e: qualified[e] for e in c.ids}, kind)
            left_out += [d for d in dropped if d.split(":")[0] not in got]
            for e, level in levels.items():
                got.setdefault(e, level)
    return got, left_out


def _levels(client: llm.Client, unit: Unit, found: dict, end: float | None) -> tuple[dict[str, str], list[str]]:
    """For entities whose target is growth against a base in their row and that have no history prior, the latest
    level copied from their documents as a prior, added to `found`; what was copied and what was left out."""
    got, left_out = _copies(client, unit, extract.bases(unit, set(found)), extract.LATEST_LEVEL, end)
    for e, level in got.items():
        found[e] = priors.growth(unit, level)
    return {e: f"{level.period}: {level.value:g}" for e, level in got.items()}, left_out


def _again(client: llm.Client, unit: Unit, found: dict, decided: dict[str, str], calls: list[plan.Call],
           texts: dict[str, list[str]], end: float | None, limit: int) -> list[plan.Call]:
    """Calls asking once more for the entities no reply gave a forecast for and that have no prior to fall back on
    (a reply cut at the output cap, a failed call), sent within the request `limit` and the clock; their replies are
    added to `texts`. An entity with a prior takes it instead (asking again never fired on the practice units)."""
    answered = {e for c in calls for text in texts.get(c.name, [])
                for e, b in reply.parse(text)[1].items() if b.get("point_forecast") is not None}
    missing = [e for e in unit.entities if e.entity_id not in answered and e.entity_id not in found]
    room = limit - client.requests
    if not missing or room <= 0:
        return []
    part = unit.model_copy(update={"entities": missing})
    again = [replace(c, name=f"a{c.name or '1-'}") for c in plan.calls(part, decided, most_calls=room)]
    print(f"  asking again for {', '.join(e.entity_id for e in missing)}", flush=True)
    got, _, _ = _run(client, again, end, limit=limit)
    texts |= got
    return [c for c in again if c.name in got]


def samples_for(calls: int, room: int = SAMPLE_CALLS) -> int:
    """The most of SAMPLES whose draws of `calls` fit SAMPLE_CALLS and the `room` left in the request budget; else 1."""
    return next((k for k in SAMPLES if k * calls <= min(SAMPLE_CALLS, room)), 1)


def answer_from(unit: Unit, calls: list[tuple[list[str], list[str]]], *, floor: bool, found: dict | None = None,
                samples: list[list[tuple[list[str], list[str]]]] | None = None,
                earlier: dict[str, extract.Level] | None = None, until: float | None = None) -> tuple[dict, dict]:
    """The answer from each call's replies (latest first), given with the entities it asks for, and what it took: the
    entities answered, the reasons written, kept and left out, the repairs. `evidence_trace` holds every reason
    written. With `found`, the computed priors are combined with the replies (`priors.combine`). With `samples` (the
    calls grouped by sample), the entity blocks are each sample's merged (`reply.merge`); the reasons are every
    call's. With `earlier` (levels copied with their dates), a yield change's intervals are widened to the least
    half-width the stated moves give (`moves`). Reasons are chosen until the clock passes `until`."""
    reasons, blocks = reply.parse_calls(calls)
    if samples and len(samples) > 1:
        blocks = reply.merge([reply.parse_calls(group)[1] for group in samples])
    if found:
        blocks = priors.combine(unit, blocks, found)
    blocks, decided = signals.apply(blocks, signals.labels(unit))
    if (half := moves.half_width(unit, earlier or {})) is not None:
        blocks = moves.widen(blocks, half)
    rows, missing = cited(unit, reply.predictions(unit, blocks))
    answer, repairs = assemble(unit, rows, floor=floor)
    kept, dropped = reply.selected(unit, reasons, answer["entity_predictions"], until)
    if kept:
        answer["submitted_reasons"] = kept
    answer["evidence_trace"] = "\n".join(f"{r.get('premise', '')} {r.get('mechanism', '')} {r.get('implication', '')}"
                                         for r in reasons)[:20_000]
    return answer, {"answered": len(rows), "least_half_width": half,
                    "reasons": {"written": len(reasons), "kept": len(kept), "dropped": dropped},
                    "repairs": missing + repairs
                    + [f"{e}: label set by its filing's going-concern or rating disclosure" for e in decided]}


def build_answer(unit: Unit, *, deadline_at: float | None = None, floor: bool = False) -> dict:
    client = llm.Client()
    t0 = time.monotonic()
    try:
        found = priors.priors(unit)
    except Exception as exc:  # noqa: BLE001 -- a unit is answered without priors rather than not at all
        print(f"  PRIORS failed: {exc}", flush=True)
        found = {}
    # what code decides is worked out first and shown to the model, so its reasons explain the submitted forecast: a
    # prior that decides its point (a copied growth level needs its copy first), a label the filings decide
    copied, left_out = _levels(client, unit, found, deadline_at)
    decided = {e: priors.decided(unit, f) for e, f in found.items() if f.decides}
    decided |= {e: f'label {label}, from the filing: "{passage}"' for e, (label, passage) in signals.findings(unit).items()}
    # the answer calls take what the copies left, less the earlier-level copy's share
    keep = EARLIER_REQUESTS if moves.starts(unit) else 0
    room = llm.MAX_REQUESTS - client.requests - keep
    try:
        calls = plan.calls(unit, decided, most_calls=room)
    except Exception as exc:  # noqa: BLE001 -- the rows alone still give the model something to answer from
        print(f"  PLAN {type(exc).__name__}: {exc}; asking with the entity rows only", flush=True)
        calls = plan.rows_only(unit, decided)
    # samples steady the model's points; when every entity's point is decided by its prior one draw is enough
    every_decided = len(found) == len(unit.entities) and all(f.decides for f in found.values())
    drawn = 1 if every_decided else samples_for(len(calls), room)
    groups = [calls] + [[replace(c, name=f"s{k}-{c.name}") for c in calls] for k in range(2, drawn + 1)]
    every = [c for g in groups for c in g]
    print(f"model={client.model}  endpoint={client.url}  tokens={render.counting()}  {len(calls)} call(s) x {drawn} "
          f"sample(s), local {time.monotonic() - t0:.1f}s", flush=True)
    # an entity no sample answers takes its prior; one with none is asked once more, else takes the floor
    texts, failures, _ = _run(client, every, deadline_at, limit=llm.MAX_REQUESTS - keep)
    again = _again(client, unit, found, decided, every, texts, deadline_at, llm.MAX_REQUESTS - keep)
    groups[0] = groups[0] + again
    every += again
    earlier, missed = (_copies(client, unit, moves.starts(unit), extract.EARLIER_LEVEL, deadline_at) if texts
                       else ({}, []))
    answer, done = answer_from(unit, [(c.ids, texts[c.name]) for c in every if c.name in texts], floor=floor,
                               found=found, samples=[[(c.ids, texts[c.name]) for c in g if c.name in texts]
                                                     for g in groups], earlier=earlier,
                               until=None if deadline_at is None else deadline_at + REASONS_AFTER)
    for line in done["repairs"]:
        print(f"  REPAIR {line}", flush=True)
    answer["notes"] = {"agent": "solution.pipeline", "pipeline": "v4.12", "tokens": render.counting(),
                       "calls": len(calls), "samples": drawn,
                       "failed_calls": failures, "roster": len(unit.entities),
                       "priors": {e: f.method for e, f in found.items()},
                       "levels": {"copied": copied, "left out": left_out},
                       "moves": {"copied": {e: f"{level.period}: {level.value:g}" for e, level in earlier.items()},
                                 "left out": missed}} | done
    path = TRACE_DIR / f"{unit.task_id}-{time.strftime('%Y%m%dT%H%M%S')}.json"
    try:  # the trace is for development; a disk that refuses it must not cost the answer
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"task_id": unit.task_id, "model": client.model, "url": client.url,
                                    "calls": client.trace}, indent=1, default=str))
    except OSError as exc:
        path = f"not written ({exc})"
    print(f"  trace: {path} ({len(client.trace)} requests)\n  budget: {client.spent()}", flush=True)
    violations = preflight(unit, answer)
    if reasons := [v for v in violations if v.startswith("g1_schema: submitted_reasons")]:
        answer.pop("submitted_reasons")
        answer["notes"]["reasons_left_out"] = reasons
        violations = preflight(unit, answer)
    for violation in reasons + violations:
        print(f"  GATE {violation}", flush=True)
    return answer


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tag = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--tag=")), "dev")
    out = REPO / ".local/runs" / tag
    for unit_dir in args:
        unit = load_unit(pathlib.Path(unit_dir))
        started = time.time()
        answer = build_answer(unit)
        out.mkdir(parents=True, exist_ok=True)
        (out / pathlib.Path(unit_dir).name).mkdir(parents=True, exist_ok=True)
        (out / pathlib.Path(unit_dir).name / "answer.json").write_text(json.dumps(answer, indent=1))
        print(f"wrote {out / pathlib.Path(unit_dir).name}/answer.json in {time.time() - started:.0f}s", flush=True)
