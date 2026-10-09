"""The corpus's dated table rows as data (v5.0, `.local/plans/2026-10-05-v5-design.md`).

A table is a header line of cells followed by rows whose first cell is a date (`2024-10 | 0.31 | 0.22`), cells
separated by `|` or tabs; the header names each row's cells, lowercased. A vintage table is a table whose columns are
`as_of_<release date>`: one reference period per row, its estimate as published on each release. Rows dated after the
cutoff, and columns headed by a date after it, are not history: the model sees them, a computation does not. Every value
keeps the span of its row, so a computation can cite the rows it used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from solution.types import Unit

DATE = re.compile(r"^\d{4}-\d{2}(?:-\d{2})?$")
SEPARATOR = re.compile(r"^:?-{2,}:?$")
NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[+-]?\.\d+")
VINTAGE = re.compile(r"as_of_(\d{4}-\d{2}-\d{2})")
KEY_DATE = re.compile(r"\d{4}[-_]\d{2}[-_]\d{2}")
#: Where a value stands: its document and the [start, end) of its row in the original text.
Span = tuple[str, int, int]

_ROWS: dict[str, list[Row]] = {}


@dataclass(frozen=True)
class Row:
    span: Span
    date: str
    cells: dict[str, str]


@dataclass(frozen=True)
class Point:
    date: str
    value: float
    span: Span


def cells(line: str) -> list[str] | None:
    """A table line's cells, split on `|` (else tabs); None for a line that is not one."""
    sep = "|" if "|" in line else "\t" if "\t" in line else None
    return [c.strip() for c in line.split(sep)] if sep else None


def late(date: str, cutoff: str) -> bool:
    """Whether a date (a month counts from its first day; `_` may stand for `-`) falls after the cutoff."""
    date = date.replace("_", "-")
    return (date if len(date) == 10 else date + "-01") > cutoff


def rows(unit: Unit) -> list[Row]:
    """Every dated table row of the unit's documents before the cutoff, in document order."""
    if unit.task_id not in _ROWS:
        cutoff = str(unit.cutoff_date)[:10]
        out = []
        for d in unit.docs:
            head, at = None, 0
            for line in d.text.split("\n"):
                start, at = at, at + len(line) + 1
                row = cells(line)
                if row is None or len(row) < 2:
                    head = None
                elif all(SEPARATOR.match(c) or not c for c in row):
                    continue
                elif not DATE.match(row[0]):
                    head = [c.lower() for c in row]
                elif head and len(row) <= len(head) and not late(row[0], cutoff):
                    named = {h: v for h, v in zip(head[1:], row[1:])
                             if v and not any(late(x, cutoff) for x in KEY_DATE.findall(h))}
                    if named:
                        out.append(Row((d.doc_id, start, start + len(line.rstrip())), row[0], named))
        _ROWS[unit.task_id] = out
    return _ROWS[unit.task_id]


def number(cell: str) -> float | None:
    """A cell's value as a number; None for a missing value ("--") or text."""
    m = NUMBER.fullmatch(cell.strip())
    return float(m.group(0).replace(",", "")) if m else None


def series(unit: Unit, doc_id: str, key: str) -> list[Point]:
    """The numeric values of `key` in a document's dated rows, in date order."""
    found = [Point(r.date, v, r.span) for r in rows(unit)
             if r.span[0] == doc_id and key in r.cells and (v := number(r.cells[key])) is not None]
    return sorted(found, key=lambda p: p.date)


def vintages(unit: Unit, doc_id: str) -> dict[str, list[Point]]:
    """A document's vintage table: per reference period, its published estimates by release date (Point.date), in
    release order; a release where the period had no estimate yet is left out."""
    out: dict[str, list[Point]] = {}
    for r in rows(unit):
        if r.span[0] != doc_id:
            continue
        found = [Point(m.group(1), v, r.span) for k, c in r.cells.items()
                 if (m := VINTAGE.fullmatch(k)) and (v := number(c)) is not None]
        if found:
            out[r.date] = sorted(found, key=lambda p: p.date)
    return out
