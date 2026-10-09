"""The prompts of an answer call (lab v4.12's texts, ported 2026-10-04).

The system text gives the role and the job (a careful financial analyst; evidence only, as of the cutoff; verbatim
quotes; what the target is, by target type; what a good answer is), then the output format: <reasons> with one
reason for each entity and one for each driver several entities share, at least 3, each its citations first, then
premise, mechanism, implication and an optional scope (the reasoning behind the forecasts lives here); then <forecasts>, one block per entity with its claims, its forecast and
interval; and a filled sample. Documents are shown as their original text (`render`), so a quote is an exact copy of
a passage or of a whole table row. The user text holds the material only (the documents, each with its manifest label
as `about`, the task's facts (target, label definitions, cutoff and resolution dates) and its prompt, the call's
entities, with a forecast code decides on its entity's line) and a closing line naming the entities in order. Apart from
the quote rule, the task facts and the decided forecasts, the texts are lab v4.12's.
"""

from __future__ import annotations

import re

from solution import render
from solution.types import Entity, Unit

SYSTEM = """You are a careful financial analyst. For each entity in a task's table you forecast one target as of the task's cutoff date, and you give the reasons and the evidence behind the forecasts.

Use ONLY the documents and the entity rows provided: no outside knowledge, and nothing about events after the cutoff. Where the documents differ from what you remember, the documents are right. Every quote you give is copied verbatim, character for character, from the documents, or for a claim from the entity's own row.

{target_rule}

A good answer has forecasts close to what then happens; {level} intervals that hold the realized value {level} of the time and are no wider than that needs; claims that are exact passages about their entity; and reasons that name the real drivers, rest on the passages they cite, and agree with the forecasts.

OUTPUT FORMAT
Reply with two sections and nothing else: <reasons>, then <forecasts>.
<reasons> holds one reason for each entity, on what drives its own forecast, and one for each driver that moves several entities, about all of them: at least 3 in all, each a different one, strongest first. Each reason is a <reason n="..."> block with these lines, in this order:
citation: {quote}, from the documents, not the entity rows. One to three: the passages the reason rests on.
premise: the fact those passages state, in one sentence, with any value from the entity rows it relies on.
mechanism: why that fact moves the forecasts, with any arithmetic.
implication: what it implies for the forecasts and intervals, naming the entities and the values forecast, as their entity blocks give them.
scope: the ids of the entities the reason is about, comma-separated; left out when it is about every entity.
<forecasts> holds one <entity id="..."> block per entity, in the order the user lists them, with these lines, in this order:
claim: {quote}, from a document about that entity or about the market (its <doc> tag's about names the entity, or is "market"); or a value from the entity's own row in <entities>, as "key: value" [task]. One to three: the passages its forecast rests on.{label_line}
point_forecast: {point}
int_lo and int_hi: the bounds of a {level} prediction interval: wide enough that the realized value falls inside it {level} of the time, and no wider.
Each reason stays under 2,000 characters.
Every line takes one of the sample's shapes; nothing goes outside the two sections.
Sample (illustrative entities and figures, complete and valid):
{example}"""
#: What the forecast fields mean, by target type.
TARGET_RULE = {"classification": "The target is a label: one of the task's labels, which the task defines. With it you give a point forecast of the quantity the label is about, as the task defines it, in its units.",
               "regression": "The target is a value, in the units the task gives.",
               "ranking": "The target ranks the entities: you forecast the metric the task ranks them by, in its units, and the entities are ranked by your forecasts."}
#: The point_forecast line, by target type.
POINT = {"classification": "your forecast of the quantity the label is about, in the units the task gives.",
         "regression": "your forecast of the value, in the units the task gives.",
         "ranking": "your forecast of the metric the entities are ranked by, in the units the task gives."}
#: How a quote is written: documents are shown as their original text.
QUOTE = "an exact copy of a passage (at least 8 words) or of a whole table row, in double quotes"
#: The sample's quotes: industry orders (a table row), EXAMPLE_A's guidance, past monthly moves, EXAMPLE_B's shipments.
QUOTES = ('"2025-03 | 6.0 | 118"',
          '"we expect April shipments of 120,000 to 130,000 units"',
          '"over the past twelve months, monthly shipments moved between a 12% decline and a 10% rise"',
          '"Shipments in March declined to 84,000 units from 91,000"')
EXAMPLE = """<reasons>
<reason n="1">
citation: {orders}
premise: Industry orders rose 6% in March.
mechanism: Orders lead shipments by about a month, so March's rise lifts April shipments at both companies.
implication: Both ship more in April than their March trend alone gives: EXAMPLE_A about 126 thousand units, EXAMPLE_B about 82 thousand.
</reason>
<reason n="2">
citation: {guidance}
premise: EXAMPLE_A guides April shipments to 120 to 130 thousand units, after shipping 118 thousand in March.
mechanism: Guidance sits above March's 118 thousand, and with orders up 6% April lands near its midpoint.
implication: EXAMPLE_A ships about 126 thousand units in April.
scope: EXAMPLE_A
</reason>
<reason n="3">
citation: {fell}
premise: EXAMPLE_B's shipments fell to 84 thousand units in March.
mechanism: Nothing points to a rebound, so April stays near March less the usual drift; the lift in orders offsets only part of it.
implication: EXAMPLE_B ships about 82 thousand units in April.
scope: EXAMPLE_B
</reason>
<reason n="4">
citation: {moves}
premise: Over the past twelve months, monthly shipments moved between a 12% fall and a 10% rise.
mechanism: A one-month forecast can miss by about that much, so a 90% interval spans about 10% either side of the point.
implication: EXAMPLE_A's interval is 112 to 138 around 126 thousand, and EXAMPLE_B's 72 to 92 around 82 thousand.
</reason>
</reasons>
<forecasts>
<entity id="EXAMPLE_A">
claim: {guidance}
claim: "march_shipments_k: 118" [task]{label_line_a}
point_forecast: 126
int_lo: 112
int_hi: 138
</entity>
<entity id="EXAMPLE_B">
claim: {fell}{label_line_b}
point_forecast: 82
int_lo: 72
int_hi: 92
</entity>
</forecasts>"""
USER = """<documents>
{corpus}
</documents>

<task>
{facts}
{prompt}
</task>

<entities>
{entities}
</entities>

{decided}Answer for these entities, in this order: {ids}. Reply with <reasons>, then <forecasts>, in the output format."""
#: Said when some entity's line carries a forecast computed from the documents, which the answer submits.
DECIDED = ("Where an entity's line gives a decided forecast, the answer submits that forecast: give it in the entity's "
           "block, and write the reasons that explain it from the documents.\n\n")
#: A document's opening tag in the rendered corpus.
DOC = re.compile(r'<doc id="D(\d+)"')


def entity_lines(entities: list[Entity], decided: dict[str, str] | None = None) -> str:
    """The entities as plain lines: `id — name | feature: value; ...`, and `| decided forecast: ...` where code
    decides it; an entity without a name takes its first `*name*` feature, else shows its id alone."""
    lines = []
    for e in entities:
        name = e.name or next((str(v) for k, v in e.features.items() if "name" in k), "")
        head = f"{e.entity_id} — {name}" if name else e.entity_id
        line = f"{head} | " + "; ".join(f"{k}: {v}" for k, v in e.features.items() if k != "corpus_ref")
        lines.append(line + (f" | decided forecast: {decided[e.entity_id]}" if decided and e.entity_id in decided else ""))
    return "\n".join(lines)


def about(unit: Unit, corpus: str) -> str:
    """The rendered corpus with each document's manifest label in its <doc> tag: "market" for a shared document, else
    the ids of the entities it is about."""
    out, last = [], 0
    for m in DOC.finditer(corpus):
        d = unit.docs[int(m.group(1)) - 1]
        label = "market" if d.shared else ", ".join(d.entity_ids) if d.entity_ids else None
        out += [corpus[last:m.end()], f' about="{label}"' if label else ""]
        last = m.end()
    return "".join(out) + corpus[last:]


def example(unit: Unit) -> str:
    orders, guidance, moves, fell = QUOTES
    labels = unit.target_labels
    # the sample's labels show the line's shape only: its entities' shipments say nothing about what a label means
    marks = {"label_line_a": "", "label_line_b": ""}
    if labels:
        marks = {"label_line_a": f"\nlabel: {labels[0]}", "label_line_b": f"\nlabel: {labels[-1]}"}
    return EXAMPLE.format(orders=orders, guidance=guidance, moves=moves, fell=fell, **marks)


def system(unit: Unit) -> str:
    labels = unit.target_labels
    return SYSTEM.format(quote=QUOTE, level=f"{unit.interval_level:.0%}",
                         target_rule=TARGET_RULE[unit.target_type], point=POINT[unit.target_type],
                         label_line=f"\nlabel: one of {', '.join(labels)}." if labels else "", example=example(unit))


def facts(unit: Unit) -> str:
    """The task's own facts, which its prompt may not restate: the target, what each label asserts, the cutoff and the
    resolution date."""
    out = [f"target: {unit.target_name} ({unit.target_type})"]
    out += [f"label {k}: {v}" for k, v in unit.label_assertions.items()]
    out.append(f"cutoff date: {unit.cutoff_date}" + (f"; resolution date: {unit.resolution_date}" if unit.resolution_date else ""))
    return "\n".join(out)


def user(unit: Unit, ids: list[str], docs: frozenset[str] | None, decided: dict[str, str] | None = None,
         caps: dict[str, int] | None = None) -> str:
    """The user text of a call answering `ids` and showing the documents in `docs` (all of them when None), each fitted
    to its cap in `caps`, with the forecasts code decides for some of them (`decided`)."""
    shown = {e: d for e, d in (decided or {}).items() if e in ids}
    return USER.format(corpus=about(unit, render.rendered(unit, docs, caps)), facts=facts(unit), prompt=unit.prompt,
                       entities=entity_lines([e for e in unit.entities if e.entity_id in ids], shown),
                       decided=DECIDED if shown else "", ids=", ".join(ids))
