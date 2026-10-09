"""Computed priors: for an entity whose own history is a dated series in the corpus, a forecast from that history
(`solution.forecast`), joined to the model's (design C, `.local/plans/2026-10-05-v5-design.md`).

The series is found from the data, never from unit names:
- a vintage table in one of the entity's own documents, when a value of its row names a reference period of the
  table and another a release date after the cutoff: the period's next revision;
- else a column of dated rows named as the entity is (its name, or a text value of its row such as a series id), in
  its own or the market-wide documents; or, in its own documents only, the one column whose name's words all appear
  in the target's name. No match, or a tie, gives no prior.
A target named as a change, growth or return (`Unit.change_target`) is the series' change from its last value to the
resolution date (in basis points when the target says so and the series is in percent); any other, the series' next
value.

Two rules join a prior to the model's answer (`combine`; measured by ablation,
`.local/analysis/2026-10-07-simplification/ABLATIONS.md`): the point is the model's, unless the prior decides it (a
revision, or a growth from a copied level, where the data's point measured far better than the model's); the interval
is the prior's band around the point. An entity the model leaves out takes the prior, and every prior cites the last
row it used.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import replace

from solution import data, extract, forecast
from solution.types import Entity, Unit

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PERCENT = {"pct", "percent", "percentage"}
PERIOD = re.compile(r"^\d{4}-\d{2}$")


def _finite(value: object, fallback: float) -> float:
    return value if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) else fallback


def _words(name: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", name.lower().replace("%", " pct ")) if w}


def _own(unit: Unit, entity: Entity) -> list[str]:
    return [d.doc_id for d in unit.docs if entity.entity_id in (d.entity_ids or [])]


def _market(unit: Unit) -> list[str]:
    return [d.doc_id for d in unit.docs if d.shared]


def _named(entity: Entity) -> set[str]:
    values = [entity.name or ""] + [v for v in entity.features.values() if isinstance(v, str)]
    return {v.strip().lower() for v in values if v.strip()}


def _column(unit: Unit, entity: Entity, rows: list[data.Row]) -> tuple[str, str] | None:
    """The (document, column) holding the entity's series, or None when no column, or more than one, matches."""
    own, market = _own(unit, entity), _market(unit)
    columns = {(r.span[0], k) for r in rows if r.span[0] in own + market for k, v in r.cells.items()
               if data.number(v) is not None}
    named = [c for c in columns if c[1] in _named(entity)]
    if len(named) == 1:
        return named[0]
    target = _words(unit.target_name)
    fitting = [c for c in columns if c[0] in own and _words(c[1]) and _words(c[1]) <= target]
    best = max((len(_words(c[1])) for c in fitting), default=0)
    top = [c for c in fitting if len(_words(c[1])) == best]
    return top[0] if len(top) == 1 else None


def _revision(unit: Unit, entity: Entity) -> forecast.Forecast | None:
    cutoff = str(unit.cutoff_date)[:10]
    values = [v for v in entity.features.values() if isinstance(v, str)]
    releases = [v for v in values if DATE.match(v) and v > cutoff]
    for doc in _own(unit, entity):
        table = data.vintages(unit, doc)
        period = next((v for v in values if PERIOD.match(v) and v in table), None)
        if period and releases and table[period]:
            latest = table[period][-1]
            # the model called revision directions at chance in every version; the revision prior's point decides
            found = forecast.revision(table, period, latest.date, latest.value, min(releases), unit.interval_level)
            return replace(found, decides=True)
    return None


def _in_percent(entity: Entity, column: str, last: float) -> bool:
    """Whether a series is in percent, so a target in basis points scales it by 100: its column says so, or a percent
    field of the entity's row (`start_yield_pct`) holds a value of its size."""
    if _words(column) & PERCENT:
        return True
    return any(_words(k) & PERCENT and isinstance(v, int | float) and v and 0.5 <= last / v <= 2
               for k, v in entity.features.items())


def prior(unit: Unit, entity: Entity, rows: list[data.Row]) -> forecast.Forecast | None:
    """The entity's computed forecast, or None when its history is not found."""
    if (found := _revision(unit, entity)) is not None:
        return found
    column = _column(unit, entity, rows)
    if column is None:
        return None
    points = data.series(unit, *column)
    if len(points) < forecast.MIN_FIT + 2:
        return None
    if unit.change_target and unit.resolution_date:
        bps = "bps" in unit.target_name or "bps" in str(entity.features.get("unit", ""))
        scale = 100.0 if bps and _in_percent(entity, column[1], points[-1].value) else 1.0
        steps = forecast.steps_between(points, points[-1].date, str(unit.resolution_date))
        return forecast.change(points, steps, scale, unit.interval_level)
    return forecast.next_value(points, unit.interval_level)


def growth(unit: Unit, level: extract.Level) -> forecast.Forecast:
    """The growth a level implies against its base (in percent when the target is), as a prior with no band of its
    own (no history measures its error) that decides the point: moved back a quarter on the filings' own figures, the
    level carried forward beat the latest growth rate carried forward (the model's habit), MAE 25.4 against 40.6."""
    point = (level.value - level.base) / abs(level.base) * (100.0 if _words(unit.target_name) & PERCENT else 1.0)
    return forecast.Forecast(point, point, point, f"the latest reported level ({level.period}: {level.value:g}) against "
                             f"the row's {level.base_field} ({level.base:g})", [], [], quote=level.quote,
                             decides=True)


def priors(unit: Unit) -> dict[str, forecast.Forecast]:
    """Each entity's prior; an entity whose series a method cannot read (a shape no practice unit has) gets none, and
    the others keep theirs."""
    rows = data.rows(unit)
    out = {}
    for e in unit.entities:
        try:
            found = prior(unit, e, rows)
        except (ValueError, ZeroDivisionError, ArithmeticError, statistics.StatisticsError) as exc:
            print(f"  PRIOR {e.entity_id}: none ({type(exc).__name__}: {exc})", flush=True)
            continue
        if found is not None:
            out[e.entity_id] = found
    return out


def _direction(unit: Unit, f: forecast.Forecast, point: float) -> str | None:
    """A revision's direction: the point's side of the latest estimate, else the prior's lean ("" with neither); None
    for any other target."""
    if f.base is None or not {"up", "down"} <= set(unit.target_labels):
        return None
    return "up" if point > f.base else "down" if point < f.base else f.label or ""


def decided(unit: Unit, f: forecast.Forecast) -> str:
    """How a prior that decides its entity's point is shown to the model: the point, a revision's direction, and the
    method that computed them."""
    direction = _direction(unit, f, f.point)
    return f"{f'label {direction}, ' if direction else ''}point_forecast {f.point:.6g}, from {f.method}"


def combine(unit: Unit, blocks: dict[str, dict], found: dict[str, forecast.Forecast]) -> dict[str, dict]:
    """The model's entity blocks with each entity that has a prior joined to it: the model's point unless the prior
    decides it or the model gave none; the prior's band around the point (for a ranking target its width centred on
    the point, the prior's own offsets carrying its method's drift; else moved to the point and still covering the
    prior's band; a prior with no band takes the model's band moved to the point and stretched to hold the model's
    point); a revision's direction from the point's side of the latest estimate, else the prior's lean; a claim of the
    last row the prior used."""
    out = dict(blocks)
    for e, f in found.items():
        b = dict(out.get(e) or {"claims": []})
        model = b.get("point_forecast")
        point = f.point if model is None or f.decides else model
        if model is None:
            lo, hi = f.lo, f.hi
        elif unit.target_type == "ranking":
            half = (f.hi - f.lo) / 2
            lo, hi = point - half, point + half
        elif f.hi > f.lo:
            lo, hi = min(f.lo, point + f.lo - f.point), max(f.hi, point + f.hi - f.point)
        else:
            shift = point - model
            lo = min(_finite(b.get("int_lo"), model) + shift, f.point, model)
            hi = max(_finite(b.get("int_hi"), model) + shift, f.point, model)
        b["point_forecast"], b["int_lo"], b["int_hi"] = point, lo, hi
        if (direction := _direction(unit, f, point)) is not None:
            b["label"] = direction or b.get("label")
        elif f.label and not b.get("label"):
            b["label"] = f.label
        if f.anchor:
            doc, start, end = f.anchor
            b["spans"] = [*b.get("spans", []), {"doc_id": doc, "span_start": start, "span_end": end}]
        if f.quote:
            b["claims"] = [*b.get("claims", []), {"node_id": "", "quote": f.quote, "claim": f.quote}]
        out[e] = b
    return out
