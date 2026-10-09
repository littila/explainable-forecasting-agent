"""The least half-width of a yield change's 90% interval, from the largest move the unit's documents state.

A unit's documents may state earlier levels of an entity's yield (a rates snapshot's trailing context: "at the
2022-06-16 close the 2-year yield was 3.14 percent"). Copied with their dates (`extract.EARLIER_LEVEL`), each gives a
move up to the cutoff; the largest, scaled to the target's horizon by the square root of time, sets the least
half-width hypot(A, K x move), and with no move stated it is A. The model writes about +/-10 bps for this target
whatever the documents say, and an instruction did not change that. A and K were fitted on the FOMC validation set
(`.local/validation/fomc`, 58 meetings, all resolved before the public units' cutoffs).
"""

from __future__ import annotations

import math
import re

from solution import extract, forecast
from solution.types import Unit

TARGET = re.compile(r"yield_change_bps")
START = re.compile(r"^start_\w*_pct$")
A = 28.0
K = 1.05
#: The horizons A was fitted on, in weekdays (the validation meetings'); outside them A scales with the square root of
#: the horizon instead of being carried to a horizon it was never fitted on.
HORIZONS = (27, 63)


def starts(unit: Unit) -> dict[str, tuple[str, float]]:
    """The entities whose earlier levels are asked for, each with its row's level at the cutoff."""
    if not TARGET.search(unit.target_name):
        return {}
    out = {}
    for e in unit.entities:
        found = next(((k, float(v)) for k, v in e.features.items()
                      if START.match(k) and isinstance(v, int | float) and not isinstance(v, bool)), None)
        if found:
            out[e.entity_id] = found
    return out


def half_width(unit: Unit, earlier: dict[str, extract.Level]) -> float | None:
    """The least half-width for the unit's intervals, in basis points; None when the target is not a yield change."""
    if not TARGET.search(unit.target_name) or not unit.resolution_date:
        return None
    cutoff = str(unit.cutoff_date)[:10]
    horizon = forecast.weekdays(cutoff, str(unit.resolution_date)[:10])
    moves = [abs(level.base - level.value) * 100.0 * math.sqrt(horizon / span) for level in earlier.values()
             if (span := forecast.weekdays(level.period, cutoff)) > 0]
    a = A * math.sqrt(horizon / min(max(horizon, HORIZONS[0]), HORIZONS[1])) if horizon > 0 else A
    return math.hypot(a, K * max(moves, default=0.0))


def widen(blocks: dict[str, dict], half: float) -> dict[str, dict]:
    """The entity blocks with each interval at least `half` either side of its point."""
    out = dict(blocks)
    for e, b in blocks.items():
        point = b.get("point_forecast")
        if point is None:
            continue
        lo, hi = b.get("int_lo"), b.get("int_hi")
        out[e] = {**b, "int_lo": min(point - half, point if lo is None else lo),
                  "int_hi": max(point + half, point if hi is None else hi)}
    return out
