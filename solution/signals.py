"""Signals an entity's own filings state that decide a credit-event label, applied after the model's answer.

A filing that discloses substantial doubt about the issuer's ability to continue as a going concern is the first
solvency signal the credit-event task names, and it is read as an event even when management says its plans
alleviate the doubt. A disclosed investment-grade rating with no such doubt is read as no event: the one-year default
rate of investment-grade issuers is a fraction of a percent. The model weighs both against losses and charges and
calls them either way from run to run.
"""

from __future__ import annotations

import re

from solution.types import Unit

LABELS = {"credit_event", "no_event"}
GOING_CONCERN = re.compile(r"substantial doubt (?:about|regarding|as to) (?:the Company|the company|our|its)\W.{0,40}?"
                           r"ability to continue as a going concern", re.DOTALL)
NEGATED = re.compile(r"\b(?:no|not|whether)\b[^.]{0,100}$", re.IGNORECASE)
RATING = re.compile(r"credit ratings?\b.{0,400}", re.IGNORECASE | re.DOTALL)
INVESTMENT_GRADE = re.compile(r"\b(?:Aaa|Aa[1-3]|A[1-3]|Baa[1-3]|AAA|AA[+-]?|A[+-]|BBB[+-]?)(?![\w+-])")
SPECULATIVE = re.compile(r"\b(?:Ba[1-3]|B[1-3]|Caa[1-3]?|Ca|BB[+-]?|B[+-]|CCC[+-]?)(?![\w+-])")


def _text(unit: Unit, entity_id: str) -> str:
    return " ".join(re.sub(r"\s+", " ", d.text) for d in unit.docs if entity_id in (d.entity_ids or ()))


def doubt(text: str) -> str | None:
    """The first passage concluding substantial doubt about the issuer's ability to continue as a going concern."""
    return next((m.group(0) for m in GOING_CONCERN.finditer(text)
                 if not NEGATED.search(text[max(0, m.start() - 120):m.start()])), None)


def investment_grade(text: str) -> str | None:
    """The first credit-rating passage with an investment grade, when no rating passage carries a speculative one."""
    windows = [m.group(0) for m in RATING.finditer(text)]
    if any(SPECULATIVE.search(w) for w in windows):
        return None
    return next((w[:INVESTMENT_GRADE.search(w).end()] for w in windows if INVESTMENT_GRADE.search(w)), None)


def findings(unit: Unit) -> dict[str, tuple[str, str]]:
    """The label each entity's filings decide, with the passage that decides it, for a credit-event target; empty for
    any other target."""
    if set(unit.target_labels) != LABELS:
        return {}
    out = {}
    for e in unit.entities:
        text = _text(unit, e.entity_id)
        if passage := doubt(text):
            out[e.entity_id] = ("credit_event", passage)
        elif passage := investment_grade(text):
            out[e.entity_id] = ("no_event", passage)
    return out


def labels(unit: Unit) -> dict[str, str]:
    return {e: label for e, (label, _) in findings(unit).items()}


def apply(blocks: dict[str, dict], decided: dict[str, str]) -> tuple[dict[str, dict], list[str]]:
    """The entity blocks with each decided label set; a probability on the other side of one half is mirrored with its
    interval, so the forecast agrees with its label. Which entities changed."""
    out, changed = dict(blocks), []
    for e, label in decided.items():
        b = out.get(e)
        if not b or b.get("label") == label:
            continue
        b = {**b, "label": label}
        p = b.get("point_forecast")
        if p is not None and 0.0 <= p <= 1.0 and (p < 0.5) == (label == "credit_event"):
            lo, hi = b.get("int_lo"), b.get("int_hi")
            b["point_forecast"] = 1.0 - p
            if lo is not None and hi is not None:
                b["int_lo"], b["int_hi"] = 1.0 - hi, 1.0 - lo
        out[e] = b
        changed.append(e)
    return out, changed
