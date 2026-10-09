"""The model's structured reply read into reasons and forecasts (lab v4.3 on; ported with v4.10, 2026-10-04).

A reply has two delimited sections, every value on its own labelled line:

    <reasons>
    <reason n="1">                      -> submitted_reasons[]
    citation: "<quote>"                 -> .citations[]: placed in the documents (`cite.place`)
    premise: ... / mechanism: ... / implication: ... / scope: ID, ID
    </reason>
    </reasons>
    <forecasts>
    <entity id="ID">                    -> entity_predictions[] row
    claim: "<quote>", or "key: value" [task] -> .claims[]: placed in the entity's documents or its own task row
    reasoning: ... / label: ... / point_forecast: ... / int_lo: ... / int_hi: ...
    </entity>
    </forecasts>

A run of several calls has one reasons section per call; the unit keeps REASONS_MAX of them from every call's reasons
by `selected`, which checks each with the grader's own rules (`baselines/guardrails_example/citation_rail.py`).
"""

from __future__ import annotations

import re
import time
from collections import Counter

from baselines.guardrails_example.citation_rail import (
    CorpusDoc,
    check_submitted_reasons,
    reasons_judged,
)
from solution import cite
from solution.types import Unit

REASON = re.compile(r"^\s*<reason\b[^>]*>\s*$", re.IGNORECASE)
#: A reply's reasons section opens; a run of several calls has one per call.
SECTION = re.compile(r"^\s*<reasons>\s*$", re.IGNORECASE)
ENTITY = re.compile(r'^\s*<entity\s+id\s*=\s*"?([^">\s]+)"?\s*>\s*$', re.IGNORECASE)
CLOSE = re.compile(r"^\s*</(reason|entity)>\s*$", re.IGNORECASE)
FIELD = re.compile(r"^\s*(premise|mechanism|implication|scope|citation|claim|reasoning|label|point_forecast|int_lo|int_hi)\s*:\s*"
                   r"(.*)$", re.IGNORECASE)
QUOTE = re.compile(r'^["“](?P<quote>.+?)["”]\s*(?:\[(?P<id>D\d+(?:\.\d+)?|task)\])?\s*\.?$')
#: A quote followed by a bracket that is not a line id (`[D7.10-11, approx lines]`) or by a note: the quote runs to its
#: last closing mark, and is found by its text.
LOOSE = re.compile(r'^["“](?P<quote>.+)["”]\s*(?:\[(?P<id>[^\]]*)\])?[^"”]*$')
LINE_ID = re.compile(r"D\d+(?:\.\d+)?|task")
NUMBER = re.compile(r"[-+−]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")
#: A field written as an element (`<premise>...</premise>`), read as `premise: ...`.
ELEMENT = re.compile(r"^(\s*)<(premise|mechanism|implication|scope|citation|claim|reasoning|label|point_forecast|int_lo|int_hi)>",
                     re.IGNORECASE)
ELEMENT_END = re.compile(r"</(premise|mechanism|implication|scope|citation|claim|reasoning|label|point_forecast|int_lo|int_hi)>\s*$",
                         re.IGNORECASE)
REASONS_MAX = 3
REQUIRED = ("premise", "mechanism", "implication")
#: Findings that rule a reason out on its own: a malformed reason, a deny-list hit (it stops the unit's whole
#: reasoning).
REFUSED = ("reasons_shape", "deny_list")
#: Findings that rule a citation out of its reason: one the judge would not see, one over 8,000 characters (it would
#: keep the whole reason from being judged).
UNSHOWN = ("reason_citation", "cap_citation_chars")


def quoted(text: str) -> dict | None:
    m = QUOTE.match(text.strip()) or LOOSE.match(text.strip())
    if not m:
        return None
    node = m.group("id") if m.group("id") and LINE_ID.fullmatch(m.group("id")) else ""
    return {"node_id": node, "quote": m.group("quote"), "claim": m.group("quote")}


def number(text: str) -> float | None:
    m = NUMBER.search(text)
    return float(m.group(0).replace(",", "").replace("−", "-")) if m else None


def parse(text: str) -> tuple[list[dict], dict[str, dict]]:
    """The reasons in order, and each entity block's claims and forecast fields by entity id. A field may be written as
    an element (`<premise>...</premise>`); a line that continues a premise, mechanism, implication or an entity's
    reasoning is kept with it. Each reason records its reasons section (`_section`) and its place in it
    (`_position`)."""
    reasons: list[dict] = []
    entities: dict[str, dict] = {}
    reason: dict | None = None
    entity: dict | None = None
    last: tuple[dict, str] | None = None
    section = position = 0
    for line in text.splitlines():
        line = ELEMENT_END.sub("", ELEMENT.sub(r"\1\2: ", line))
        if SECTION.match(line):
            section, position = section + 1, 0
        elif REASON.match(line):
            position += 1
            reason, entity, last = {"citations": [], "_section": section, "_position": position}, None, None
            reasons.append(reason)
        elif m := ENTITY.match(line):
            reason, last = None, None
            entity = entities.setdefault(m.group(1), {"claims": []})
        elif CLOSE.match(line):
            reason = entity = last = None
        elif m := FIELD.match(line):
            key, value = m.group(1).lower(), m.group(2).strip()
            last = None
            if reason is not None:
                if key == "citation" and (q := quoted(value)):
                    reason["citations"].append(q)
                elif key == "scope":
                    reason["scope"] = [x.strip() for x in value.split(",") if x.strip()]
                elif key in REQUIRED:
                    reason[key], last = value, (reason, key)
            elif entity is not None:
                if key == "claim" and (q := quoted(value)):
                    entity["claims"].append(q)
                elif key == "reasoning":
                    entity["reasoning"], last = value, (entity, "reasoning")
                elif key == "label":
                    entity["label"] = value.strip().strip(".")
                elif key in ("point_forecast", "int_lo", "int_hi"):
                    entity[key] = number(value)
        elif last and line.strip() and not line.strip().startswith(("<", "```")):
            block, key = last
            block[key] = f"{block[key]} {line.strip()}".strip()
    return reasons, entities


def merge(samples: list[dict[str, dict]]) -> dict[str, dict]:
    """Entity blocks from several samples of the same calls, as one: per entity the median sample's block (its point
    and claims, the lower one of an even count) among the samples with the most common label (the earliest sample's
    on a tie), with an interval spanning all of those samples' intervals. Samples that disagree are uncertainty, and a
    90% interval pays twenty times the distance it misses by against once its width."""
    out = {}
    for e in dict.fromkeys(e for s in samples for e in s):
        rows = [s[e] for s in samples if e in s and s[e].get("point_forecast") is not None]
        if not rows:
            out[e] = next(s[e] for s in samples if e in s)
            continue
        if labels := [r["label"] for r in rows if r.get("label")]:
            label = Counter(labels).most_common(1)[0][0]
            rows = [r for r in rows if r.get("label") == label]
        rows = sorted(rows, key=lambda r: r["point_forecast"])
        b = dict(rows[(len(rows) - 1) // 2])
        for k, pick in (("int_lo", min), ("int_hi", max)):
            if bounds := [r[k] for r in rows if r.get(k) is not None]:
                b[k] = pick(bounds)
        out[e] = b
    return out


def predictions(unit: Unit, blocks: dict[str, dict]) -> dict[str, dict]:
    """The entity blocks with a point forecast or a label, by entity in roster order, as `pipeline.cited` reads them;
    a block with a label and no point keeps its label and takes the floor's point (`assemble`)."""
    return {e.entity_id: {"point_forecast": b.get("point_forecast"), "int_lo": b.get("int_lo"), "int_hi": b.get("int_hi"),
                          "claims": b["claims"], "spans": b.get("spans", []),
                          **({"label": b["label"]} if b.get("label") else {}),
                          **({"reasoning": [b["reasoning"]]} if b.get("reasoning") else {})}
            for e in unit.entities if (b := blocks.get(e.entity_id)) and (b.get("point_forecast") is not None or b.get("label"))}

def parse_calls(calls: list[tuple[list[str], list[str]]]) -> tuple[list[dict], dict[str, dict]]:
    """Each call's replies (latest first: a corrected reply, then the one it corrects), given with the entities the call
    asks for, parsed: the reasons of each call's latest reply in call order, each with its call (`_call`) and the call's
    entities (`_ids`); per entity, the block of the latest reply that gives a forecast, else its latest block. A block
    with a forecast is never replaced by a later call's (a call asking again only fills what is missing)."""
    reasons: list[dict] = []
    blocks: dict[str, dict] = {}
    for k, (ids, texts) in enumerate(calls, 1):
        parsed = [parse(t) for t in texts]
        reasons += [r | {"_call": k, "_ids": ids} for r in parsed[0][0]]
        for e in ids:
            found = [bs[e] for _, bs in parsed if e in bs]
            best = next((b for b in found if b.get("point_forecast") is not None), found[0] if found else None)
            if best is not None and (e not in blocks or blocks[e].get("point_forecast") is None):
                blocks[e] = best
    return reasons, blocks


def _found(unit: Unit, quote: dict, about: list[str]) -> dict | None:
    """A reason citation placed in the documents of the entities it is about first, in their order, else in every
    document: a passage several filings share is cited from the one the reason is about."""
    for got in [cite.place(unit, e, quote) for e in about] + [cite.place(unit, None, quote)]:
        if got and got[0]["doc_id"] != cite.TASK:
            return {k: got[0][k] for k in ("doc_id", "span_start", "span_end")}
    return None


def _reason(unit: Unit, r: dict) -> tuple[dict, list[str]]:
    """A parsed reason in the answer's form, unnumbered, and what was left out of it: [task] citations and citations
    found nowhere."""
    left = []
    if task := [q for q in r["citations"] if q["node_id"] == cite.TASK]:
        left.append(f"{len(task)} [task] citation(s) left out (a reason cites the corpus only)")
    corpus = [q for q in r["citations"] if q["node_id"] != cite.TASK]
    about = r.get("scope") or r.get("_ids") or []
    cites = [c for q in corpus if (c := _found(unit, q, about))]
    if len(cites) < len(corpus):
        left.append(f"{len(corpus) - len(cites)} citation(s) found nowhere")
    ids = {e.entity_id for e in unit.entities}
    item = {"reason_id": "", "premise": r["premise"], "mechanism": r["mechanism"], "answer_implication": r["implication"]}
    if scope := [e for e in r.get("scope", []) if e in ids]:
        item["scope"] = {"entities": scope}
    if cites:
        item["citations"] = [dict(t) for t in dict.fromkeys(tuple(c.items()) for c in cites)]
    return item, left


def _overlaps(a: dict, b: dict) -> bool:
    return any(x["doc_id"] == y["doc_id"] and x["span_start"] < y["span_end"] and y["span_start"] < x["span_end"]
               for x in a.get("citations", []) for y in b.get("citations", []))


def _corpus(corpus: dict[str, CorpusDoc], items: list[dict]) -> dict[str, CorpusDoc]:
    """The documents `items` cite: all a reason check needs, and far less to read than the corpus."""
    return {c["doc_id"]: corpus[c["doc_id"]] for r in items for c in r.get("citations", []) if c["doc_id"] in corpus}


def _checked(unit: Unit, item: dict, rows: list[dict], corpus: dict[str, CorpusDoc]) -> tuple[dict | None, list[str]]:
    """A reason with each citation the grader would not show, or that is over its cap (UNSHOWN), left out; None when the
    grader refuses the reason on its own (REFUSED). What was left out."""
    left = []
    shown = [c for c in item.get("citations", [])
             if not {f.code for f in check_submitted_reasons(
                 {"entity_predictions": rows, "submitted_reasons": [item | {"citations": [c]}]},
                 _corpus(corpus, [{"citations": [c]}]), unit.cutoff_date)} & set(UNSHOWN)]
    if len(shown) < len(item.get("citations", [])):
        left.append(f"{len(item['citations']) - len(shown)} citation(s) left out (not shown, or over 8,000 characters)")
        item = {f: v for f, v in item.items() if f != "citations"} | ({"citations": shown} if shown else {})
    found = {f.code for f in check_submitted_reasons({"entity_predictions": rows, "submitted_reasons": [item]},
                                                     _corpus(corpus, [item]), unit.cutoff_date)}
    if refused := sorted(found & set(REFUSED)):
        return None, left + [f"refused ({', '.join(refused)})"]
    return item, left


def selected(unit: Unit, reasons: list[dict], rows: list[dict], until: float | None = None) -> tuple[list[dict], list[str]]:
    """Up to REASONS_MAX of every call's reasons (`parse_calls`) as `submitted_reasons`, and why each other one was left
    out. A reason the grader's check refuses on its own (REFUSED) is out, and a citation it would not show or that is over
    its cap (UNSHOWN) is dropped from its reason.
    The rest rank cited before uncited (the judge grades grounding on the passages cited), then by how much of the
    roster they are about (their scope, else their call's entities), then their place in their call, then call order;
    each is added while the grader would judge it (`reasons_judged`: within the caps, not a duplicate), unless it cites
    a passage a reason of another call already chosen cites. Reasons are checked in rank order and only until
    REASONS_MAX are kept or the clock passes `until`, each check reading only the documents the reasons cite: the
    grader's check reads every document it is given, which on a corpus of large filings takes seconds a call."""
    corpus = {d.doc_id: CorpusDoc(d.doc_id, d.text, str(d.doc_date)) for d in unit.docs}
    roster = {e.entity_id for e in unit.entities}
    dropped, ranked = [], []
    for k, r in enumerate(reasons, 1):
        if missing := [f for f in REQUIRED if not r.get(f)]:
            dropped.append(f"reason {k}: no {', '.join(missing)}")
            continue
        about = len(roster & set(r.get("scope") or r.get("_ids") or roster))
        ranked.append(((-about, r.get("_position", k), r.get("_call", 0), k), k, r, None))
    ranked.sort(key=lambda x: x[0])
    out: list[dict] = []
    calls: list[int] = []
    # cited reasons first; one none of whose citations is placed and shown waits with the uncited, already checked
    waiting = [x for x in ranked if not x[2]["citations"]]
    for cited_pass in (True, False):
        queue = [x for x in ranked if x[2]["citations"]] if cited_pass else sorted(waiting, key=lambda x: x[0])
        for key, k, r, item in queue:
            call = key[2]
            if len(out) == REASONS_MAX:
                dropped.append(f"reason {k}: beyond the first {REASONS_MAX}")
                continue
            if until is not None and time.monotonic() > until:
                dropped.append(f"reason {k}: not checked, out of time")
                continue
            if item is None:
                item, left = _reason(unit, r)
                checked, more = _checked(unit, item, rows, corpus)
                dropped += [f"reason {k}: {x}" for x in left + more]
                if checked is None:
                    continue
                item = checked
                if cited_pass and not item.get("citations"):
                    waiting.append((key, k, r, item))
                    continue
            if any(c != call and _overlaps(item, o) for c, o in zip(calls, out)):
                dropped.append(f"reason {k}: cites a passage a reason of another call already chosen cites")
                continue
            trial = out + [item | {"reason_id": f"r{len(out) + 1}"}]
            plan = reasons_judged({"entity_predictions": rows, "submitted_reasons": trial}, _corpus(corpus, trial),
                                  unit.cutoff_date)
            if not plan[-1]["judged"]:
                dropped.append(f"reason {k}: not judged ({plan[-1]['why']})")
                continue
            out, calls = trial, calls + [call]
    return out, dropped
