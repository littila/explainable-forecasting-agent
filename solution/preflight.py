"""Run the scorer's mechanical gates locally, before an answer is shipped.

`checks.py` answers "does this forecast look sensible" -- a constant vector, a suspicious scale, a
too-narrow band. Useful, and entirely separate from the question this module answers: "will the
harness admit this unit at all". A unit can pass every check in `checks.py` and still score
`W = -0.27` because one entity's `interval.level` is 0.9000001.

The gates replicated here are the ones whose verdict is a pure function of the answer, the task and
the corpus -- `g0_integrity`, `g1_schema`, `g2_cutoff_resource`, and everything in
`align_predictions` plus the embargo that `g3_domain_semantics` runs before faithfulness. The
faithfulness threshold itself is NOT replicated: it needs the pinned NLI ensemble, it is the one
gate a local judge can disagree with the graders about, and `assemble` cannot repair it anyway.

So a clean preflight means: every failure left is a judgement about our evidence, not a mistake in
our plumbing. That is the only claim it makes.
"""

from __future__ import annotations

import json
import math
import pathlib

import jsonschema

from solution.types import Unit

#: The submission image does not install `qfbench2-common` -- it would pull numpy and scipy for
#: one JSON file, and image pull time is charged to the unit clock. So the schema is vendored
#: beside this module and the package is preferred where it exists. Drift between the two is what
#: would make this dangerous, so `tests/test_preflight.py` fails if the copies differ, and a
#: missing copy raises here rather than quietly skipping the gate it is supposed to replicate.
VENDORED = pathlib.Path(__file__).parent / "analysis.schema.json"


def _schema() -> dict:
    try:
        from qfbench2_common.taskcard import schema_path
        return json.loads(pathlib.Path(schema_path("analysis.schema.json")).read_text())
    except ImportError:
        if not VENDORED.exists():
            raise RuntimeError(
                f"neither qfbench2_common nor {VENDORED} is available, so g1_schema cannot "
                "be replicated; refusing to report a clean preflight") from None
        return json.loads(VENDORED.read_text())


_SCHEMA = _schema()


def _g1(answer: dict) -> list[str]:
    validator = jsonschema.Draft202012Validator(_SCHEMA)
    return [f"g1_schema: {'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
            for e in sorted(validator.iter_errors(answer), key=lambda e: list(e.path))]


def _g3_alignment(unit: Unit, answer: dict) -> list[str]:
    """`align_predictions`, restated. It raises on the first failure; we want all of them."""
    out: list[str] = []
    roster = [e.entity_id for e in unit.entities]

    declared = answer.get("target_type")
    if declared is not None and declared != unit.target_type:
        out.append(f"TARGET_TYPE_MISMATCH: answer says {declared!r}, task says {unit.target_type!r}")

    rows = answer.get("entity_predictions")
    if not isinstance(rows, list) or not rows:
        return out + ["SCHEMA_INVALID: entity_predictions is empty or not a list"]

    seen: dict[str, dict] = {}
    for row in rows:
        eid = row.get("entity_id") if isinstance(row, dict) else None
        if not isinstance(eid, str) or not eid:
            out.append("SCHEMA_INVALID: a row carries no usable entity_id")
        elif eid in seen:
            out.append(f"ENTITY_DUPLICATE: {eid} appears more than once")
        elif eid not in roster:
            out.append(f"ENTITY_UNKNOWN: {eid} is not on the roster")
        else:
            seen[eid] = row
    for eid in roster:
        if eid not in seen:
            out.append(f"ENTITY_MISSING: roster entity {eid} has no row")

    ranks: list[int] = []
    for eid in roster:
        row = seen.get(eid)
        if row is None:
            continue
        out += _row_gates(unit, eid, row)
        if isinstance(row.get("rank"), int) and not isinstance(row.get("rank"), bool):
            ranks.append(row["rank"])
        elif row.get("rank") is not None:
            out.append(f"SCHEMA_INVALID: {eid} reports a non-integer rank")

    if ranks and len(ranks) != len(roster):
        out.append(f"SCHEMA_INVALID: rank on {len(ranks)} of {len(roster)} entities is not a ranking")
    elif ranks and sorted(ranks) != list(range(1, len(roster) + 1)):
        out.append(f"SCHEMA_INVALID: ranks {sorted(ranks)} are not a permutation of 1..{len(roster)}")
    return out


def _row_gates(unit: Unit, eid: str, row: dict) -> list[str]:
    out: list[str] = []
    interval = row.get("interval")
    if not isinstance(interval, dict):
        out.append(f"INTERVAL_INVALID: {eid} carries no interval object")
    elif "level" not in interval:
        out.append(f"INTERVAL_LEVEL_MISMATCH: {eid} carries no interval level")
    else:
        lo, hi, level = interval.get("lo"), interval.get("hi"), interval.get("level")
        if not _is_finite(level) or float(level) != float(unit.interval_level):
            out.append(f"INTERVAL_LEVEL_MISMATCH: {eid} reports {level!r}, "
                       f"unit requires {unit.interval_level}")
        if not _is_finite(lo) or not _is_finite(hi):
            out.append(f"NONFINITE_VALUE: {eid} interval [{lo!r}, {hi!r}]")
        elif lo > hi:
            out.append(f"INTERVAL_INVALID: {eid} interval is inverted ({lo} > {hi})")

    label = row.get("label")
    if unit.target_type == "classification":
        if not isinstance(label, str) or not label:
            out.append(f"LABEL_INVALID: {eid} carries no label on a classification unit")
        elif unit.target_labels and label not in unit.target_labels:
            out.append(f"LABEL_INVALID: {eid} label {label!r} is outside {unit.target_labels}")

    point = row.get("point_forecast")
    if point is None:
        if unit.target_type in ("regression", "ranking"):
            out.append(f"SCHEMA_INVALID: {eid} carries no point_forecast on a {unit.target_type} unit")
    elif not _is_finite(point):
        out.append(f"NONFINITE_VALUE: {eid} point_forecast is {point!r}")

    claims = row.get("claims")
    if not isinstance(claims, list) or not claims:
        out.append(f"CITATION_MALFORMED: {eid} carries no claims[]")
    else:
        out += _citation_gates(unit, eid, claims)
    return out


def _citation_gates(unit: Unit, eid: str, claims: list) -> list[str]:
    by_id = {d.doc_id: d for d in unit.docs}
    out: list[str] = []
    for i, claim in enumerate(claims):
        where = f"{eid} claim[{i}]"
        if not isinstance(claim, dict):
            out.append(f"CITATION_MALFORMED: {where} is not an object")
            continue
        if claim.get("doc_id") == "task":
            # the task table (scorer 5.2.x): a span inside the citing entity's own row
            start, end = claim.get("span_start"), claim.get("span_end")
            row = unit.task_rows.get(eid)
            if not isinstance(start, int) or not isinstance(end, int) or row is None or not row[0] <= start < end <= row[1]:
                out.append(f"TASK_SPAN: {where} cites task [{start!r}:{end!r}] outside {eid}'s row {row}")
            continue
        doc = by_id.get(claim.get("doc_id"))
        if doc is None:
            out.append(f"CITATION_UNRESOLVED: {where} cites {claim.get('doc_id')!r}, "
                       "which is not in the corpus")
            continue
        if doc.doc_date > unit.cutoff_date:
            out.append(f"EMBARGO: {where} cites {doc.doc_id} dated {doc.doc_date}, "
                       f"after the {unit.cutoff_date} cutoff")
        start, end = claim.get("span_start"), claim.get("span_end")
        # Not a gate: the scorer slices without bounds-checking, so a bad span is a silent empty
        # premise and a guaranteed zero for that citation rather than a refusal. Still ours to fix.
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            out.append(f"SPAN_EMPTY: {where} has offsets [{start!r}:{end!r}]")
        elif end > len(doc.text):
            out.append(f"SPAN_OUT_OF_RANGE: {where} ends at {end}, "
                       f"past {doc.doc_id}'s {len(doc.text)} chars")
    return out


def _is_finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def preflight(unit: Unit, answer: dict) -> list[str]:
    """Every mechanical reason this answer would be refused, or an empty list.

    `g0_integrity` is the caller's read of the file and `g2_cutoff_resource` is one comparison, so
    both are folded in here rather than given their own passes.
    """
    out: list[str] = []
    if not isinstance(answer, dict):
        return ["g0_integrity: answer is not a JSON object"]
    if answer.get("task_id") != unit.task_id:
        out.append(f"g2_cutoff_resource: task_id {answer.get('task_id')!r} != {unit.task_id!r}")
    return out + _g1(answer) + _g3_alignment(unit, answer)
