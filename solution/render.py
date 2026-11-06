"""The corpus as the model reads it: every document as its original text, cited by quote.

A document is shown in a <doc> tag with its title, date and source, as it was written: a table row dated after the
cutoff in a document dated before it (an auction calendar, a projection, a curve by contract month) was known on the
document's date, and the computed history reads it too (`data.rows`). An original text larger than a call can
hold is fitted: whitespace collapsed, then the windows that rank best against the task, in document order with each
gap marked (`fit`). A quote is found in the original text whatever was left out of the view (`cite.locate`).
"""

from __future__ import annotations

import functools
import re
from xml.sax.saxutils import quoteattr

from solution import retrieve
from solution.tokens import counter, estimate
from solution.types import Doc, Unit

#: A call's prompt takes at most CONTEXT_SHARE of WINDOW tokens. A document may take at most DOC_MAX of them: the
#: call's share less PROMPT_RESERVE for the prompt and an entity's small companion documents. A larger one is fitted
#: in windows of WINDOW_CHARS, or of half its cap in characters when that is smaller (so the best window fits a small
#: cap even where the text is denser than its average), and never under MIN_WINDOW_CHARS (about two sentences); each
#: gap marker is counted as GAP_TOKENS.
WINDOW = 262_144
CONTEXT_SHARE = 0.4
PROMPT_RESERVE = 8_192
DOC_MAX = int(CONTEXT_SHARE * WINDOW) - PROMPT_RESERVE
WINDOW_CHARS = 16_000
MIN_WINDOW_CHARS = 200
GAP_TOKENS = 16

_SHOWN: dict[tuple[str, str, int], str] = {}


@functools.cache
def _count():
    return counter()


def prompt_tokens(text: str) -> int:
    return _count()(text)


def counting() -> str:
    return "estimate" if _count() is estimate else "exact"


def owned(unit: Unit, entity_id: str) -> frozenset[str]:
    """The documents an entity may cite: by the unit manifest's labels, those about it and the market-wide ones; a
    document the manifest does not label is taken as market-wide."""
    return frozenset(d.doc_id for d in unit.docs
                     if d.shared or d.entity_ids is None or entity_id in d.entity_ids)


def rendered(unit: Unit, docs: frozenset[str] | None = None, caps: dict[str, int] | None = None) -> str:
    """The documents (those in `docs` when given, by doc_id) as the model reads them, D1 being the unit's first, each
    fitted to its cap in `caps` (DOC_MAX when it has none)."""
    return "\n".join(shown(unit, k, d, (caps or {}).get(d.doc_id, DOC_MAX)) for k, d in enumerate(unit.docs, 1)
                     if docs is None or d.doc_id in docs)


def shown(unit: Unit, k: int, d: Doc, cap: int = DOC_MAX) -> str:
    """A document in its <doc> tag: its text, fitted to `cap` tokens when larger."""
    key = (unit.task_id, d.doc_id, cap)
    if key not in _SHOWN:
        _SHOWN[key] = fit(d.text, retrieve.retrieval_query(unit), cap)
    return (f"<doc id=\"D{k}\" title={quoteattr(d.title)} date=\"{d.doc_date}\" source={quoteattr(d.source)}>\n"
            f"{_SHOWN[key]}\n</doc>")


def fit(text: str, query: str, cap: int = DOC_MAX) -> str:
    """`text` within `cap` tokens: as it is when it fits; else with its whitespace collapsed; else its windows
    (`windows`, sized to the cap) ranked by BM25 against `query`, kept best-first while they fit, shown in document
    order with each gap marked."""
    if prompt_tokens(text) <= cap:
        return text
    text = re.sub(r"\s+", " ", text).strip()
    tokens = prompt_tokens(text)
    if tokens <= cap:
        return text
    spans = windows(text, max(MIN_WINDOW_CHARS, min(WINDOW_CHARS, len(text) * (cap - GAP_TOKENS) // (2 * tokens))))
    parts = [text[a:b] for a, b in spans]
    cost = [prompt_tokens(x) + GAP_TOKENS for x in parts]
    ranked = retrieve.BM25(parts).ranked(query)
    order = ranked + sorted(set(range(len(parts))) - set(ranked))
    keep, used = [], 0
    for i in order:
        if used + cost[i] <= cap:
            keep.append(i)
            used += cost[i]
    out, last = [], 0
    for i in sorted(keep):
        a, b = spans[i]
        if a > last:
            out.append(f"[... {a - last:,} characters left out ...]")
        out.append(parts[i])
        last = b
    if last < len(text):
        out.append(f"[... {len(text) - last:,} characters left out ...]")
    return " ".join(out)


def windows(text: str, size: int | None = None) -> list[tuple[int, int]]:
    """`text` cut into contiguous spans of about `size` characters (WINDOW_CHARS by default), each ended at a sentence
    end in its second half when there is one, else at a space."""
    size = size or WINDOW_CHARS
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        if j < len(text):
            end = text.rfind(". ", i + size // 2, j)
            j = end + 2 if end > 0 else (text.rfind(" ", i + size // 2, j) + 1 or j)
        out.append((i, j))
        i = j
    return out
