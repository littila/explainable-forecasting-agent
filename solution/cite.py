"""Where a quote-first claim points (lab v3.16k's resolver, ported 2026-10-02; plain text since 2026-10-07).

A claim's quote is looked for in the documents the entity may cite only (its own and the shared ones; scorer 5.2.x
counts a citation of any other document as a false claim), in their original text, then by the quote's longest piece
around "...". Matching folds case, quote marks, dashes, invisible characters and whitespace runs, and may also ignore
whitespace altogether, since a filing's table cells are spaced differently from a copy of them. A quote found nowhere
is not cited.

A claim may also cite its entity's own row of the task table (`"task"`, docs/CONCEPTS.md "Citing the task table"):
the model quotes the row as the prompt shows it (`key: value`), and each quoted key whose value is the row's becomes its
own citation, that key's `"key": value` in the row's JSON. Reasons cite the corpus only.

A claim's text is the cited passage's own words (`verbatim`): a paraphrase can state a figure its passage does not
carry, which scorer 5.2.x counts as a false claim.
"""

from __future__ import annotations

import json
import re

from solution.render import owned
from solution.types import Unit

#: Characters a quote may differ in from its source without being a different quote. Zero-width spaces count as
#: spaces: EDGAR filings put them between table cells, and a model cannot see or copy them.
SAME = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
                      "\u00a0": " ", "\u200b": " ", "\u200c": " ", "\u200d": " ", "\u2060": " ", "\ufeff": " "})
#: The shortest opening part of a quote that may stand for it when the whole quote is not in the document.
QUOTE_MIN = 40
#: The shortest quote, without its whitespace, that the whitespace-stripped match may place.
STRIPPED_MIN = 8
#: The reserved doc_id of the task table.
TASK = "task"
#: The longest claim text, in characters. The scorer counts a claim over 400 judge tokens as false, and figure-dense
#: text runs about two characters a token.
CLAIM_MAX = 600
#: The fewest words of a quote the claim keeps as its text; a shorter one without a figure gives way to its passage.
QUOTE_WORDS = 8

_FOLDED: dict[tuple[str, int], tuple[str, list[int]]] = {}
_STRIPPED: dict[tuple[str, int], tuple[str, list[int]]] = {}


def folded(text: str) -> tuple[str, list[int]]:
    """`text` lowercased, with quote marks and dashes unified and whitespace runs collapsed, and for each folded
    character the index of the original character it came from."""
    out, where = [], []
    for i, ch in enumerate(text.translate(SAME).lower()):
        if ch.isspace():
            if out and out[-1] == " ":
                continue
            ch = " "
        out.append(ch)
        where.append(i)
    return "".join(out), where


def _folded_doc(unit: Unit, k: int) -> tuple[str, list[int]]:
    if (unit.task_id, k) not in _FOLDED:
        _FOLDED[unit.task_id, k] = folded(unit.docs[k].text)
    return _FOLDED[unit.task_id, k]


def stripped_text(unit: Unit, k: int) -> tuple[str, list[int]]:
    """Document k's folded text with all whitespace removed (cached), and each character's index in the original."""
    if (unit.task_id, k) not in _STRIPPED:
        hay, where = _folded_doc(unit, k)
        keep = [i for i, ch in enumerate(hay) if not ch.isspace()]
        _STRIPPED[unit.task_id, k] = ("".join(hay[i] for i in keep), [where[i] for i in keep])
    return _STRIPPED[unit.task_id, k]


def locate(unit: Unit, doc_ref: str, quote: str, opening: bool = True) -> tuple[str, int, int] | None:
    """A quoted passage's (doc_id, start, end) in the original text of the document shown as `doc_ref` (D1 is the
    unit's first document): found as written, or folded, or with all whitespace removed, or (with `opening`) as the
    longest opening part of at least QUOTE_MIN characters the document holds; None when not even that is there."""
    k = int(doc_ref[1:]) - 1
    if not 0 <= k < len(unit.docs) or len(quote) < 8:
        return None
    text = unit.docs[k].text
    at = text.find(quote)
    if at >= 0:
        return unit.docs[k].doc_id, at, at + len(quote)
    hay, where = _folded_doc(unit, k)
    needle = folded(quote)[0].strip()
    at = hay.find(needle)
    if at < 0:
        bare = re.sub(r"\s+", "", needle)
        squeezed, spots = stripped_text(unit, k)
        at = squeezed.find(bare) if len(bare) >= STRIPPED_MIN else -1
        if at >= 0:
            return unit.docs[k].doc_id, spots[at], spots[at + len(bare) - 1] + 1
    # a quote that runs past its passage or was altered after its start: the longest opening part the document
    # holds, cut at a word boundary and at least QUOTE_MIN characters long; the span then runs as far as the quote is
    # long, within the line and to the end of a word, so it takes in what the quote altered
    whole = len(needle)
    if at < 0 and opening:
        # an opening found means every shorter one is found too, so the longest is found by bisection over the word
        # ends, not by searching the document once a word
        cuts = [p for p in range(QUOTE_MIN, len(needle) - 1) if needle[p] == " "]
        lo, hi = 0, len(cuts)
        while lo < hi:
            mid = (lo + hi) // 2
            lo, hi = (mid + 1, hi) if hay.find(needle[:cuts[mid]]) >= 0 else (lo, mid)
        if lo:
            needle = needle[:cuts[lo - 1]]
            at = hay.find(needle)
    if at < 0:
        return None
    start, end = where[at], where[at + len(needle) - 1] + 1
    if len(needle) < whole:
        line_end = text.find("\n", end)
        line_end = len(text) if line_end < 0 else line_end
        end = max(end, min(start + whole, line_end))
        while end < line_end and not text[end].isspace():
            end += 1
    return unit.docs[k].doc_id, start, end


def verbatim(text: str, start: int, end: int, quote: str = "") -> str:
    """The claim for a citation of text[start:end]: the quote as the passage has it when the passage holds it
    (folded match) and it has at least QUOTE_WORDS words or a figure, else the passage (a short quote without a
    figure, such as a row's label, can say nothing on its own); whitespace runs as one space, at most CLAIM_MAX
    characters cut at a space so that no figure is cut."""
    passage = text[start:end]
    needle = folded(quote)[0].strip() if len(quote.split()) >= QUOTE_WORDS or re.search(r"\d", quote) else ""
    hay, where = folded(passage)
    at = hay.find(needle) if needle else -1
    if at >= 0:
        passage = passage[where[at]:where[at + len(needle) - 1] + 1]
    passage = " ".join(passage.split())
    if len(passage) > CLAIM_MAX:
        cut = passage.rfind(" ", 0, CLAIM_MAX + 1)
        passage = passage[:cut if cut > 0 else CLAIM_MAX]
    return passage


def _same_value(quoted: str, value: object) -> bool:
    """Whether a quoted value is the row's value: equal as a number, else as text (quotes and case aside)."""
    quoted = quoted.strip().rstrip(".").strip().strip('"').strip()
    if isinstance(value, bool) or value is None:
        return quoted.lower() == json.dumps(value)
    if isinstance(value, (int, float)):
        try:
            return float(quoted.replace(",", "")) == float(value)
        except ValueError:
            return False
    return quoted.casefold() == str(value).strip().casefold()


def task_spans(unit: Unit, entity_id: str, quote: str) -> list[tuple[int, int]]:
    """Where a quote of an entity's row stands in the task table: one [start, end) per quoted key whose value is the
    row's, at the row's own `"key": value`. The prompt shows a row as `key: value; key: value`; a JSON-style copy
    (`"key": value, ...`) is read too. A key quoted with a value the row does not have cites nothing."""
    if entity_id not in unit.task_rows or not quote.strip():
        return []
    start, end = unit.task_rows[entity_id]
    text = unit.task_table[start:end]
    row = json.loads(text)
    out = []
    for key, value in row.items():
        m = re.search(rf'(?:^|[;|,{{]\s*)"?{re.escape(key)}"?\s*:\s*(.+?)\s*(?=;|\||,\s*"?[A-Za-z_][\w]*"?\s*:|}}|$)', quote)
        if m and _same_value(m.group(1), value):
            pair = json.dumps({key: value}, ensure_ascii=False, separators=(", ", ": "))[1:-1]
            if (at := text.find(pair)) >= 0:
                out.append((start + at, start + at + len(pair)))
    return sorted(out)


def _task(unit: Unit, entity_id: str | None, quote: str) -> list[dict]:
    return [{"doc_id": TASK, "span_start": start, "span_end": end, "claim": verbatim(unit.task_table, start, end)}
            for start, end in (task_spans(unit, entity_id, quote) if entity_id else [])]


def place(unit: Unit, entity_id: str | None, claim: dict) -> list[dict] | None:
    """A quote-first claim's citations, in the documents the entity may cite (every document when `entity_id` is None:
    a reason's citation is not bound to an entity): a `[task]` quote in the entity's own row; else the quote's span in
    the original text of the first such document holding it whole (`locate`), else of the first holding its opening,
    in its own words; else the same for the quote's longest piece around "..." or "…"; else, for a claim, the
    entity's own row (`task_spans`). None when it is found nowhere."""
    quote = claim.get("quote", "")
    if claim.get("node_id") == TASK:
        return _task(unit, entity_id, quote) or None
    own = [f"D{k}" for k, d in enumerate(unit.docs, 1) if entity_id is None or d.doc_id in owned(unit, entity_id)]
    pieces = sorted((x.strip() for x in re.split(r"\.{3,}|…", quote)), key=len, reverse=True)
    tries = [(piece, opening) for piece in dict.fromkeys([quote.strip(), *pieces]) if folded(piece)[0].strip()
             for opening in (False, True)]
    for piece, opening in tries:
        for d in own:
            if found := locate(unit, d, piece, opening):
                doc, start, end = found
                text = unit.docs[int(d[1:]) - 1].text
                return [{"doc_id": doc, "span_start": start, "span_end": end, "claim": verbatim(text, start, end)}]
    return _task(unit, entity_id, quote) or None
