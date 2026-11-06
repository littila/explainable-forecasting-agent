"""Levels of an entity's quantity copied from its documents, for priors and intervals the model's forecast is combined
with (design C, `.local/plans/2026-10-05-v5-design.md` section 8).

Two kinds of copy. The latest level of a growth target's quantity: an entity qualifies when it has no history series
(`solution.priors`), its target is growth (`target_name`), and its row gives the base the growth is measured against
(a numeric `prior_*` field). The earliest earlier level, with its date, of a quantity whose row gives its level at the
cutoff (`solution.moves`). One call per batch of qualifying entities, showing their documents as the answer call
does, asks for a value, its period and a passage quoted exactly; it forecasts nothing. Code keeps a value only when the
quote is found in the entity's own documents, the value is one of the quote's numbers and differs from the row's, and
for an earlier level when its period is a date before the cutoff.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass

from solution import cite, plan, prompt, render, reply
from solution.types import Unit

GROWTH = re.compile(r"(?<![a-z])growth(?![a-z])")
BASE = re.compile(r"prior_")
#: A base field named as a rate or a change is not a level a growth can be measured against.
RATE = {"growth", "change", "return", "returns", "rate", "pct", "percent", "yoy", "ratio"}
#: A copied level this far from its base (a fifth, five times) is another quantity, period or scale.
RATIO = (0.2, 5.0)
#: Reply tokens an entity's line plans on (a value, a period and one quoted passage).
PER_ENTITY = 250
SYSTEM = ("You copy figures from documents. You do not forecast, estimate or compute: every value you give is printed "
          "in the documents, and every quote is copied from them character for character.")
USER = """<documents>
{corpus}
</documents>

<task>
{prompt}
</task>

<entities>
{entities}
</entities>

{request}"""
LATEST = """Each entity's row gives a base value: {bases}. For each entity, find in its documents the same quantity for the most recent period they report, later than the base's period. Reply with one line per entity, in this order: {ids}. Each line exactly:
<entity id> | value: <the number as printed> | period: <the period it is for> | quote: "<the passage that states it, copied exactly>"
Write nothing else."""
EARLIER = """Each entity's row gives its level at the cutoff: {bases}. The documents above are the sentences of the unit's documents that state a figure for a date before the cutoff. For each entity, list every level of its own quantity these sentences state, each with its date. Entities in this order: {ids}; an entity may have several lines. Each line exactly:
<entity id> | value: <the number as printed> | period: <the date, as YYYY-MM-DD> | quote: "<the words that state it, copied exactly>"
For an entity whose level no sentence states, the line is: <entity id> | none
Write nothing else."""
#: The follow-up sent once when lines are missing or fail the checks.
CORRECTION = ("Your reply needs fixing: {{issues}}. Reply again in full, one line per entity in the same format: the "
              "{which} value as printed, and the passage that states it copied exactly from the documents.")
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
SENTENCE_END = re.compile(r"(?<=[.;])\s+")
NONE = re.compile(r"^\W*([^|]*?[^\s|])\s*\|\s*none\W*$", re.IGNORECASE)
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
#: A number in a quote, with a leading-dot decimal as filings print it (".97").
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?|\.\d+")
LINE = re.compile(r'^\W*([^|]*?[^\s|])\s*\|\s*value:\s*([^|]+?)\s*\|\s*period:\s*([^|]*?)\s*\|\s*quote:\s*["“](.+)["”]\s*\W*$')


@dataclass(frozen=True)
class Kind:
    #: Call names ("e1-", ...), the request, its follow-up, and whether the period is a date before the cutoff and an
    #: entity may have none.
    prefix: str
    request: str
    correction: str
    dated: bool


LATEST_LEVEL = Kind("e", LATEST, CORRECTION.format(which="latest period's"), False)
EARLIER_LEVEL = Kind("p", EARLIER, CORRECTION.format(which="earliest earlier date's"), True)


@dataclass(frozen=True)
class Level:
    value: float
    period: str
    quote: str
    base_field: str
    base: float


def bases(unit: Unit, without: set[str]) -> dict[str, tuple[str, float]]:
    """The entities that qualify (none in `without`, those with a history prior), each with its base field and value."""
    if not GROWTH.search(unit.target_name.lower()):
        return {}
    out = {}
    for e in unit.entities:
        if e.entity_id in without:
            continue
        found = next(((k, float(v)) for k, v in e.features.items()
                      if BASE.match(k) and not RATE & set(re.split(r"[^a-z0-9]+", k.lower()))
                      and isinstance(v, int | float) and not isinstance(v, bool) and v), None)
        if found:
            out[e.entity_id] = found
    return out


def earlier_sentences(unit: Unit) -> list[str]:
    """The documents' sentences that state a figure for a date before the cutoff; table rows (lines with cells) are
    read as series and left out."""
    cutoff = str(unit.cutoff_date)[:10]
    out = []
    for d in unit.docs:
        for line in d.text.splitlines():
            if "|" in line:
                continue
            for sentence in SENTENCE_END.split(line.strip()):
                dates = ISO_DATE.findall(sentence)
                if any(x < cutoff for x in dates) and NUMBER.search(ISO_DATE.sub("", sentence)):
                    out.append(sentence)
    return list(dict.fromkeys(out))


def calls(unit: Unit, qualified: dict[str, tuple[str, float]], kind: Kind = LATEST_LEVEL,
          most_calls: int | None = None) -> list[plan.Call]:
    """The extraction calls, named "e1-", "e2-", ... (by the kind's prefix), each with its entities' documents, at most
    `most_calls` of them where the output plan allows; for a dated kind one call showing only the sentences that state
    earlier figures, none when there are none."""
    if kind.dated:
        ids = [e.entity_id for e in unit.entities if e.entity_id in qualified]
        return [plan.Call(f"{kind.prefix}1-", ids, SYSTEM, _user(unit, ids, frozenset(), qualified, kind))] if (
            ids and earlier_sentences(unit)) else []
    part = unit.model_copy(update={"entities": [e for e in unit.entities if e.entity_id in qualified]})
    every = [e.entity_id for e in part.entities]
    overhead = render.prompt_tokens(SYSTEM + _user(unit, every, frozenset(), qualified, kind))
    return [plan.Call(f"{kind.prefix}{k}-", b.entities, SYSTEM, _user(unit, b.entities, b.docs, qualified, kind, b.caps))
            for k, b in enumerate(plan.batches(part, PER_ENTITY, overhead, most_calls=most_calls), 1)]


def _user(unit: Unit, ids: list[str], docs: frozenset[str], qualified: dict[str, tuple[str, float]], kind: Kind,
          caps: dict[str, int] | None = None) -> str:
    shown = ", ".join(f"{e}: {qualified[e][0]} = {qualified[e][1]:g}" for e in ids)
    corpus = ("\n".join(earlier_sentences(unit)) if kind.dated
              else prompt.about(unit, render.rendered(unit, docs, caps)))
    return USER.format(corpus=corpus, prompt=unit.prompt,
                       entities=prompt.entity_lines([e for e in unit.entities if e.entity_id in ids]),
                       request=kind.request.format(bases=shown, ids=", ".join(ids)))


MONTHS = {m: k for k, m in enumerate(("january", "february", "march", "april", "may", "june", "july", "august",
                                         "september", "october", "november", "december"), 1)}
MONTH_YEAR = re.compile(r"\b(" + "|".join(MONTHS) + r")\b\.?\s*(?:\d{1,2},?\s*)?(\d{4})", re.IGNORECASE)


def _after(period: str, date: str) -> bool:
    """Whether a period named by month and year ends after `date` (YYYY-MM-DD), on its month's last day: a filing
    of that date cannot report it."""
    m = MONTH_YEAR.search(period)
    if not m:
        return False
    year, month = int(m.group(2)), MONTHS[m.group(1).lower()]
    return f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}" > date


def _found(unit: Unit, e: str, quote: str) -> str | None:
    """The quote as the documents print it: as given, else with inserted dollar signs and spacing folded away."""
    for q in dict.fromkeys((quote, re.sub(r"\s+", " ", quote.replace("$", "")).strip())):
        if cite.place(unit, e, {"node_id": "", "quote": q, "claim": q}) is not None:
            return q
    return None


def levels(unit: Unit, text: str, qualified: dict[str, tuple[str, float]],
           kind: Kind = LATEST_LEVEL) -> tuple[dict[str, Level], list[str]]:
    """The verified levels in a reply, and why each other line was left out."""
    cutoff = str(unit.cutoff_date)[:10]
    out, dropped = {}, []
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if not m or m.group(1) not in qualified:
            continue
        e, value, period, quote = m.group(1), reply.number(m.group(2)), m.group(3).strip(), m.group(4).strip()
        field, base = qualified[e]
        figures = [float(x.replace(",", "")) for x in NUMBER.findall(quote)]
        if value == base and base in figures and figures.index(base) > 0 and RATIO[0] < figures[0] / base < RATIO[1]:
            # a row of periods copied at the base's column: tables print the latest period first
            value, period = figures[0], "the quoted row's first column"
        own = [d.doc_date for d in unit.docs if e in (d.entity_ids or ())]
        if value is None or abs(value) not in figures:
            dropped.append(f"{e}: the value is not in the quote")
        elif not kind.dated and value != base and not RATIO[0] < value / base < RATIO[1]:
            dropped.append(f"{e}: the value is not between a fifth and five times the base; copy the same quantity")
        elif not kind.dated and own and _after(period, str(max(own))):
            dropped.append(f"{e}: the period {period} ends after the date of its documents ({max(own)}); copy the "
                           "latest period they report")
        elif (found := _found(unit, e, quote)) is None:
            dropped.append(f"{e}: the quote is not in its documents")
        elif value == base:
            dropped.append(f"{e}: the value is the base itself")
        elif kind.dated and not (DATE.match(period) and period < cutoff):
            dropped.append(f"{e}: the period is not a date before the cutoff")
        elif not (kind.dated and e in out and out[e].period <= period):
            # of several earlier levels an entity's sentences state, the oldest
            out[e] = Level(value, period, found, field, base)
    return out, dropped


def problems(unit: Unit, qualified: dict[str, tuple[str, float]], text: str, finish: str, ids: list[str],
             kind: Kind = LATEST_LEVEL) -> list[str]:
    """How an extraction reply falls short for the entities `ids`: cut off, a line missing, or a line that fails the
    checks (`levels`). For a dated kind, a line saying none is an answer."""
    asked = {e: qualified[e] for e in ids if e in qualified}
    got, dropped = levels(unit, text, asked, kind)
    seen = {d.split(":")[0] for d in dropped}
    if kind.dated:
        seen |= {m.group(1) for line in text.splitlines() if (m := NONE.match(line.strip()))}
    out = ["the reply was cut off at the length limit"] if finish == "length" else []
    return out + dropped + [f"{e}: no line" for e in asked if e not in got and e not in seen]
