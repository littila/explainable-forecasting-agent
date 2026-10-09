"""The published Track 4 submission entrypoint.

    analyze --task /input/task.json --corpus /input/corpus/ --out /output/answer.json

`analyze` is the CONTAINER COMMAND, so it arrives as the first argument after the image
reference and this parser has to consume it. An image that does not exits 2 on every unit,
before reading a single input file (SUBMISSION_CLI.md). It stays optional so the module is still
runnable by hand.

Two things make this the scored path:

**It always writes an answer.** Every path out of this function that is not a crash ends in a
file at `--out`. A unit whose model calls all failed scores whatever a floored answer is worth;
a unit with no `answer.json` scores `W = -0.27` and tells the graders nothing. So the answer is
written even when the pipeline raised, and what was lost is recorded in `notes`.

**It turns the floor on.** Development wants an empty `claims[]` to fail loudly; a scored run
wants a real citation over a certain schema failure.

The clock (spec §8a): the pipeline runs to a soft deadline `SOFT_MARGIN` before the hard one: it
stops starting calls, stops waiting for the ones in flight, and assembles what it has. This module
arms the HARD deadline with `SIGALRM`, for what the soft one cannot stop: a hang outside a call, in
parsing or assembly. `SIGALRM` interrupts the main thread whatever it is blocked in, which is the
only guarantee that survives a hang inside a C extension; a container killed at 600 s with nothing
written scores W = -0.27 and explains nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import sys
import time
import tomllib
import traceback

from solution.corpus import task_table

VERB = "analyze"

#: The wall clock this process will not outlive when the card names no time limit: below the practice cards'
#: `[agent] timeout_sec = 600`, so there is room to assemble and write after the alarm fires.
HARD_DEADLINE = float(os.environ.get("T4_HARD_DEADLINE_SEC") or 570.0)
#: How far the hard deadline sits below the card's time limit, which also counts the image pull: 30 s, or a tenth of
#: a shorter limit.
LIMIT_MARGIN = 30.0
#: The soft deadline sits this far before the hard one: time to assemble and write.
SOFT_MARGIN = 30.0


class OutOfTime(Exception):
    """The hard wall-clock guard fired."""


def _out_of_time(signum: int, frame: object) -> None:
    raise OutOfTime("alarm: the run deadline")


def hard_deadline(task_path: pathlib.Path) -> float:
    """Seconds this run may take: `T4_HARD_DEADLINE_SEC` when set; else the card's `[agent] timeout_sec` beside the task
    less LIMIT_MARGIN (a tenth of a shorter limit); else HARD_DEADLINE."""
    if os.environ.get("T4_HARD_DEADLINE_SEC"):
        return HARD_DEADLINE
    try:
        card = tomllib.loads((task_path.parent / "card.toml").read_text(encoding="utf-8"))
        limit = float(card["agent"]["timeout_sec"])
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError):
        return HARD_DEADLINE
    return limit - min(LIMIT_MARGIN, limit / 10) if limit > 0 else HARD_DEADLINE


def _from_task(task_path: pathlib.Path, why: str) -> dict:
    """The last-resort answer, for when even loading the unit failed.

    Built from `task.json` alone -- no corpus, no model, nothing that can fail a second time. Each row cites a value of
    its own task row (the one citation that needs no corpus), as the scorer indexes the task table, and takes the
    first label of the vocabulary on a classification unit, so the answer still passes the schema. `notes` names the
    cause.
    """
    task = json.loads(task_path.read_text(encoding="utf-8"))
    target = task.get("target") or {}
    labels = list(target.get("labels") or target.get("label_assertions") or [])
    table, ranges = task_table(task.get("entities", []))
    rows = []
    for e in task.get("entities", []):
        start, end = ranges.get(e.get("entity_id", ""), (0, 0))
        line = table[start:end]
        cell = next((json.dumps({k: v}, ensure_ascii=False, separators=(", ", ": "))[1:-1]
                     for k, v in e.items() if k not in ("entity_id", "corpus_ref") and v not in (None, "")), "")
        at = line.find(cell) if cell else -1
        claims = [{"doc_id": "task", "span_start": start + at, "span_end": start + at + len(cell), "claim": cell}] if at >= 0 else []
        rows.append({"entity_id": e.get("entity_id", ""),
                     "interval": {"level": float(task.get("interval_level", 0.90)), "lo": -1.0, "hi": 1.0},
                     "point_forecast": 0.0, "claims": claims, **({"label": labels[0]} if labels else {})})
    return {"task_id": task.get("task_id", task_path.parent.name),
            "schema_version": "3",
            "target_type": task.get("target_type") or target.get("type") or "classification",
            "entity_predictions": rows,
            "evidence_trace": "",
            "notes": {"agent": "solution.cli last-resort", "failed": why}}


def write(out_path: pathlib.Path, answer: dict) -> None:
    """`answer` at `out_path`, replacing what is there in one step: a reader never sees a half-written file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(answer, indent=1), encoding="utf-8")
    os.replace(tmp, out_path)


def run(task_path: pathlib.Path, corpus_dir: pathlib.Path, out_path: pathlib.Path) -> int:
    signal.signal(signal.SIGALRM, _out_of_time)
    hard = hard_deadline(task_path)
    signal.alarm(max(1, int(hard)))
    soft = time.monotonic() + hard - min(SOFT_MARGIN, hard / 10)
    unit = None
    try:
        from solution import pipeline
        from solution.corpus import load_unit

        unit = load_unit(task_path.parent, corpus_dir=corpus_dir)
        # a valid answer from the start: whatever stops the run later, a scorable file is already there
        write(out_path, pipeline.floor_answer(unit, "written before any model call"))
        answer = pipeline.build_answer(unit, deadline_at=soft, floor=True)
    except Exception:  # noqa: BLE001 -- whatever failed, an answer is still written
        traceback.print_exc()
        why = traceback.format_exc(limit=3)[-800:]
        if unit is None:
            answer = _from_task(task_path, why)
            print(f"UNIT DID NOT LOAD; wrote a refusable answer to {out_path}", file=sys.stderr)
        else:
            # The unit loaded, so the corpus is in hand and a floored answer is still gate-valid
            # -- worth whatever the judge makes of it, instead of a certain W = -0.27. The
            # forecasts are empty and `notes` says so; nothing here pretends to be a prediction.
            try:
                answer = pipeline.floor_answer(unit, why)
                print(f"AGENT FAILED; wrote a floored answer to {out_path}", file=sys.stderr)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                answer = _from_task(task_path, why)
                print(f"FLOOR FAILED; wrote a refusable answer to {out_path}", file=sys.stderr)
    finally:
        signal.alarm(0)   # the write below must never be interrupted by our own guard

    write(out_path, answer)
    print(f"wrote {out_path} ({len(answer['entity_predictions'])} entities)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QFBench 2.0 Track 4 submission (analyze).")
    parser.add_argument("verb", nargs="?", default=VERB, choices=[VERB],
                        help=f"track verb; the harness passes {VERB!r} as the container command")
    parser.add_argument("--task", required=True, type=pathlib.Path, help="path to task.json")
    parser.add_argument("--corpus", required=True, type=pathlib.Path, help="path to corpus/")
    parser.add_argument("--out", required=True, type=pathlib.Path, help="answer.json to write")
    args = parser.parse_args(argv)
    return run(args.task, args.corpus, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
