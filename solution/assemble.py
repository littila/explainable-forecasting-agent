"""Stage 3 (m8 spec §8): the replies in hand become an answer no mechanical gate can refuse.

Nothing structural is taken from the model: task_id and target_type come from the unit; one row
per roster entity, in roster order; the label coerced into the vocabulary; the point finite; the
interval at the unit's level, as given unless the scorer would refuse it; rank derived from the
points; claims as placed spans (`cite.place`), de-duplicated, none content-free or over SPAN_MAX. What
is left is `g1`'s non-empty `claims`, which the floor covers on submission. Every coercion is
returned, so none is silent.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter

from solution import cite
from solution.render import owned
from solution.types import Entity, Unit

#: The longest passage a floor citation may be, in characters.
FLOOR_MAX = 600
#: Where a passage ends: a sentence end or a line break.
PASSAGE_END = re.compile(r"(?<=[.;!?])\s+|\n+")
#: The scorer's content-free rule (5.2.2, `qfbench2_track_analysis.scoring.content_free`), copied because the image
#: does not install the scorer; `test_assemble` asserts the lists are the scorer's. A claim with no digit whose words,
#: the unit's entity names, ids and tickers set aside, are all function or meta words, and either all function words or
#: with a filler anchor among them, asserts nothing and is false.
FUNCTION_WORDS = frozenset({
    "a", "all", "also", "an", "and", "any", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "done", "each", "for", "from", "had", "has", "have", "here", "if", "in", "into", "is", "it",
    "it's", "its", "may", "might", "must", "no", "nor", "not", "of", "on", "only", "or", "over", "should", "so",
    "some", "such", "that", "the", "their", "there", "these", "this", "those", "to", "very", "was", "were", "what",
    "when", "where", "which", "while", "who", "will", "with", "would",
})
META_WORDS = frozenset({
    "available", "cite", "cited", "citing", "claim", "claims", "contains", "context", "contextual", "cutoff",
    "document", "documents", "establish", "establishes", "evidence", "excerpt", "fallback", "forecast", "forecasts",
    "grounded", "inference", "model", "model-entailed", "nearest", "only", "passage", "passages", "placeholder",
    "pre-cutoff", "prediction", "predictions", "quote", "quotes", "relevant", "retrieved", "selected", "source",
    "sources", "submitted", "support", "supporting", "supports", "top-retrieved", "used", "wording",
})
FILLER_ANCHORS = frozenset({
    "cite", "cited", "citing", "context", "contextual", "cutoff", "document", "documents", "evidence", "excerpt",
    "fallback", "inference", "model-entailed", "nearest", "passage", "passages", "placeholder", "pre-cutoff",
    "retrieved", "source", "sources", "top-retrieved", "wording",
})
#: What makes a claim contentful however few words it has: a letter outside a-z, an arrow, or a digit.
CONTENT = re.compile(r"[^\W\d_a-zA-Z]|[\u2190-\u21ff\u27f0-\u27ff\u2b00-\u2bff\u25b2\u25bc]|\d")
NOTICE = re.compile(r"\bmodel inference failed\b", re.IGNORECASE)
#: The longest span a claim may cite: past it the scorer counts the claim false, whatever it says.
SPAN_MAX = 8_000
#: The labels a floor or an invalid label falls back to, when the vocabulary names one.
NEUTRAL = ("inline", "unchanged", "flat", "none", "no_event")


#: An entity field that gives the target's latest known value.
LATEST_FIELD = re.compile(r"(latest|last)_")


def floor_point(unit: Unit, entity: Entity) -> tuple[float, str]:
    """The point an entity takes when no reply gives one: no change for a change target; else the latest value its
    own row gives of the target's quantity (a numeric `latest_*` or `last_*` field sharing a word with the target's
    name); else 0, which `assemble` replaces by the answered entities' median."""
    if unit.change_target:
        return 0.0, "no change"
    target = set(re.split(r"[^a-z0-9]+", unit.target_name.lower())) - {"", "latest", "last"}
    for k, v in entity.features.items():
        if (LATEST_FIELD.match(k) and isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
                and set(re.split(r"[^a-z0-9]+", k.lower())) & target):
            return float(v), f"the row's {k}"
    return 0.0, "no value to persist"


def _finite(value: object, fallback: float = 0.0) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def _folded_label(label: object) -> str:
    return re.sub(r"[\s\-]+", "_", str(label).strip().strip("\"'*`_ ").strip().lower())


def _label(unit: Unit, given: object, repairs: list[str], entity_id: str) -> str | None:
    """The given label in the unit's vocabulary: as written, else matched without case, quotes, markup or the
    difference between spaces, hyphens and underscores; else the vocabulary's neutral label, logged."""
    if not unit.target_labels:
        return None
    if given in unit.target_labels:
        return given  # type: ignore[return-value]
    if given is not None and (same := [x for x in unit.target_labels if _folded_label(x) == _folded_label(given)]):
        return same[0]
    neutral = next((x for x in unit.target_labels if x in NEUTRAL), unit.target_labels[0])
    repairs.append(f"{entity_id}: label {given!r} not in {unit.target_labels} -> {neutral!r}")
    return neutral


def interval(unit: Unit, point: float, lo: object, hi: object, repairs: list[str], entity_id: str) -> dict:
    """The interval as the model gave it, at the unit's level. The scorer refuses only a non-finite or inverted one, so
    a missing or non-finite bound takes the point and an inverted pair is swapped; a zero-width interval, or one that
    leaves out the point, is valid and kept as written."""
    lo_f, hi_f = _finite(lo, point), _finite(hi, point)
    if math.isnan(_finite(lo, math.nan)) or math.isnan(_finite(hi, math.nan)):
        repairs.append(f"{entity_id}: interval [{lo!r}, {hi!r}] has a missing bound -> the point")
    if hi_f < lo_f:
        repairs.append(f"{entity_id}: interval [{lo_f}, {hi_f}] inverted -> swapped")
        lo_f, hi_f = hi_f, lo_f
    return {"level": unit.interval_level, "lo": lo_f, "hi": hi_f}


def row_claim(unit: Unit, entity_id: str) -> dict | None:
    """A value of the entity's own task row as a claim (`"task"`), a numeric one first: always citable, and its figure
    is in its span."""
    entity = next(e for e in unit.entities if e.entity_id == entity_id)
    values = sorted(((k, v) for k, v in entity.features.items() if k != "corpus_ref" and v not in (None, "")),
                    key=lambda kv: not isinstance(kv[1], int | float))
    for key, value in values:
        if spans := cite.task_spans(unit, entity_id, f"{key}: {value}"):
            start, end = spans[0]
            return {"doc_id": cite.TASK, "span_start": start, "span_end": end,
                    "claim": cite.verbatim(unit.task_table, start, end)}
    return None


def _floor(unit: Unit, entity_id: str, counts: Counter) -> dict | None:
    """The passage the roster cited most among the documents this entity may cite, else the first passage of its own
    documents, then of the shared ones, that holds a figure and is at most FLOOR_MAX characters; else a value of its
    own task row (`row_claim`)."""
    scope = owned(unit, entity_id)
    for (doc, start, end), _ in counts.most_common():
        if doc in scope:
            text = next(d.text for d in unit.docs if d.doc_id == doc)
            return {"doc_id": doc, "span_start": start, "span_end": end, "claim": cite.verbatim(text, start, end)}
    mine = [d for d in unit.docs if entity_id in (d.entity_ids or ())]
    for d in mine + [d for d in unit.docs if d.doc_id in scope and d not in mine]:
        at = 0
        for m in [*PASSAGE_END.finditer(d.text), None]:
            end = m.start() if m else len(d.text)
            passage = d.text[at:end]
            if re.search(r"\d", passage) and 20 <= len(passage.strip()) <= FLOOR_MAX:
                start = at + len(passage) - len(passage.lstrip())
                return {"doc_id": d.doc_id, "span_start": start, "span_end": end,
                        "claim": cite.verbatim(d.text, start, end)}
            at = m.end() if m else len(d.text)
    return row_claim(unit, entity_id)


def _names(unit: Unit) -> tuple[str, ...]:
    names = [n for e in unit.entities for n in (e.entity_id, e.name) if n]
    names += [v for e in unit.entities for k, v in e.features.items() if k.endswith("ticker") and isinstance(v, str) and v]
    return tuple(sorted(set(names), key=len, reverse=True))


def says_nothing(text: str, names: tuple[str, ...]) -> bool:
    """The scorer's content-free test (see FUNCTION_WORDS): true for a claim that asserts nothing a passage could
    support, such as a heading that is only the entity's name."""
    for name in names:
        text = re.sub(r"(?<![\w])" + re.escape(name) + r"(?![\w])", " ", text, flags=re.IGNORECASE)
    text = NOTICE.sub(" ", text)
    if CONTENT.search(text):
        return False
    words = [w.strip("-'") for w in re.findall(r"[a-z][a-z\-']*", text.lower())]
    rest = [w for w in words if w and w not in FUNCTION_WORDS]
    return all(w in META_WORDS for w in rest) and (not rest or any(w in FILLER_ANCHORS for w in rest))


def assemble(unit: Unit, preds: dict[str, dict], *, floor: bool = False) -> tuple[dict, list[str]]:
    """The answer from per-entity predictions (rows whose claims are placed spans), and the repairs it took. The floor
    applies only to an entity left with no claim."""
    names = _names(unit)
    repairs: list[str] = []
    rows = []
    counts: Counter = Counter()
    for e in unit.entities:
        pred = preds.get(e.entity_id) or {}
        floor_value, basis = floor_point(unit, e)
        if not pred:
            repairs.append(f"{e.entity_id}: no prediction -> point {floor_value} ({basis})")
        point = _finite(pred.get("point_forecast"), floor_value)
        row: dict = {"entity_id": e.entity_id,
                     "interval": interval(unit, point, pred.get("interval_lo"), pred.get("interval_hi"),
                                           repairs, e.entity_id)}
        claims, placed = [], set()
        for c in pred.get("claims", []):
            key = (c["doc_id"], c["span_start"], c["span_end"])
            if key in placed:
                continue
            if says_nothing(c["claim"], names):
                repairs.append(f"{e.entity_id}: claim {c['claim'][:40]!r} says nothing -> dropped")
            elif c["span_end"] - c["span_start"] > SPAN_MAX:
                repairs.append(f"{e.entity_id}: claim cites {c['span_end'] - c['span_start']} characters, over "
                               f"{SPAN_MAX} -> dropped")
            else:
                placed.add(key)
                claims.append(c)
        counts.update(k for k in placed if k[0] != cite.TASK)
        row["claims"] = claims
        label = _label(unit, pred.get("label"), repairs, e.entity_id)
        if label:
            row["label"] = label
        row["point_forecast"] = point
        rows.append(row)
    # an entity no reply answered takes, from those answered, the median width and the most common label, and with no
    # value of its own to persist their median point: anything but a zero-width interval and an arbitrary label
    numbered = {e for e, p in preds.items() if math.isfinite(_finite(p.get("point_forecast"), math.nan))}
    answered = [r for r in rows if r["entity_id"] in numbered]
    for e, row in zip(unit.entities, rows):
        if e.entity_id in numbered or not answered:
            continue
        if floor_point(unit, e)[1] == "no value to persist":
            row["point_forecast"] = statistics.median(r["point_forecast"] for r in answered)
        half = statistics.median(r["interval"]["hi"] - r["interval"]["lo"] for r in answered) / 2
        row["interval"] |= {"lo": row["point_forecast"] - half, "hi": row["point_forecast"] + half}
        if not preds.get(e.entity_id, {}).get("label") and (labels := [r["label"] for r in answered if r.get("label")]):
            row["label"] = Counter(labels).most_common(1)[0][0]
        repairs.append(f"{e.entity_id}: point, interval and label from the {len(answered)} answered entities")
    for e, row in zip(unit.entities, rows):
        if row["claims"]:
            continue
        claim = _floor(unit, e.entity_id, counts) if floor else None
        if claim is None:
            repairs.append(f"{e.entity_id}: NO CITATION, claims:[] will fail g1_schema")
            continue
        row["claims"] = [claim]
        repairs.append(f"{e.entity_id}: no citation -> floor cites {claim['doc_id']} [{claim['span_start']}, "
                       f"{claim['span_end']})")
    if unit.target_type == "ranking":
        for k, row in enumerate(sorted(rows, key=lambda r: -r["point_forecast"]), 1):
            row["rank"] = k
    answer = {"task_id": unit.task_id, "schema_version": "3", "target_type": unit.target_type,
              "entity_predictions": rows, "evidence_trace": "", "notes": {}}
    return answer, repairs
