"""Forecasts computed from a series' own history in the corpus (v5.0, `.local/plans/2026-10-05-v5-design.md`).

Three forms, each a fixed menu of standard methods chosen per run by a backtest on the series' own pre-cutoff history
(nothing is fitted offline or on outcomes; `docs/TRAINING-POLICY.md`):

- `next_value`: the next observation of the series (an auction's bid-to-cover after the past auctions of its tenor).
  Menu: last value, mean of the last 3 or 6, mean or median of all. The method with the smallest mean absolute
  one-step backtest error gives the point; the interval is the point plus that method's error quantiles.
- `change`: the change of a level series over a horizon of `steps` observations (a yield's move to a later date).
  With enough history the menu (no change, drift) is backtested at that horizon; otherwise no change, with the
  observed step volatility scaled by the square root of the horizon.
- `revision`: the next published estimate of a period in a vintage table, against its latest estimate. The point is
  the latest estimate plus the median of past revisions over the same number of releases at the same age (months
  from the period to the release); the label is the direction of the point.

The interval level is the unit's: from 19 errors up (for 90%) their 5% and 95% quantiles; from fewer, the small-sample
prediction interval mean ± t(n-1) · s · sqrt(1 + 1/n), and a volatility interval takes t(n-1) for the normal quantile.
Every forecast lists the rows it used.
"""

from __future__ import annotations

import calendar
import datetime as dt
import itertools
import math
import statistics as st
from collections.abc import Callable
from dataclasses import dataclass

from solution.data import Point, Span

#: Fewest backtest errors an empirical interval rests on; with fewer, the interval comes from the step volatility.
MIN_ERRORS = 5
#: Fewest past values a mean-type method needs before it is backtested.
MIN_FIT = 3
TINY = 1e-300
MENU: dict[str, Callable[[list[float]], float]] = {
    "last value": lambda xs: xs[-1],
    "mean of the last 3": lambda xs: st.mean(xs[-3:]),
    "mean of the last 6": lambda xs: st.mean(xs[-6:]),
    "mean of all": st.mean,
    "median of all": st.median,
}
#: The `change` menu: the forecast change over `steps` from the levels seen so far.
CHANGE_MENU: dict[str, Callable[[list[float], int], float]] = {
    "no change": lambda past, steps: 0.0,
    "drift": lambda past, steps: st.mean(b - a for a, b in itertools.pairwise(past)) * steps,
}


@dataclass(frozen=True)
class Forecast:
    point: float
    lo: float
    hi: float
    method: str
    errors: list[float]
    #: The spans of the rows the forecast used.
    lines: list[Span]
    label: str | None = None
    #: The row of the last value the forecast starts from, and for a revision the estimate it is measured against.
    anchor: Span | None = None
    base: float | None = None
    #: A passage quoted exactly that states the value the forecast starts from (a level copied from a filing).
    quote: str | None = None
    #: Whether this forecast's point is the answer's, whatever the model's (`priors.combine`).
    decides: bool = False


def quantile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    i = p * (len(xs) - 1)
    k = math.floor(i)
    return xs[k] + (xs[min(k + 1, len(xs) - 1)] - xs[k]) * (i - k)


def _ibeta(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function I_x(a, b), by its continued fraction (modified Lentz)."""
    if x <= 0.0 or x >= 1.0:
        return 0.0 if x <= 0.0 else 1.0
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _ibeta(b, a, 1.0 - x)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(400):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1.0 / (d if abs(d) > TINY else TINY)
        c = 1.0 + num / (c if abs(c) > TINY else TINY)
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def t_quantile(p: float, df: int) -> float:
    """The `p` quantile (p >= 0.5) of Student's t with `df` degrees of freedom, by bisection on its distribution
    function."""
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2
        if 1 - 0.5 * _ibeta(df / 2, 0.5, df / (df + mid * mid)) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _band(center: float, errors: list[float], level: float) -> tuple[float, float]:
    """`center` plus a `level` prediction interval for the next error. With enough errors for that level (19 for a 90%
    interval: the next error falls outside the range of n errors 2/(n+1) of the time), their empirical quantiles; with
    fewer, the small-sample interval: mean ± t(n-1) · s · sqrt(1 + 1/n)."""
    n = len(errors)
    tail = (1 - level) / 2
    if n >= math.ceil(2 / (1 - level) - 1e-9) - 1:
        return center + quantile(errors, tail), center + quantile(errors, 1 - tail)
    if n < 2:
        return center, center
    half = t_quantile(1 - tail, n - 1) * st.stdev(errors) * math.sqrt(1 + 1 / n)
    return center + st.mean(errors) - half, center + st.mean(errors) + half


def _spread(diffs: list[float], level: float) -> float:
    """The half-width of a `level` interval for one step, from the observed steps: t(n-1) · s."""
    return t_quantile(0.5 + level / 2, len(diffs) - 1) * st.stdev(diffs) if len(diffs) > 1 else 0.0


def next_value(points: list[Point], level: float) -> Forecast:
    xs = [p.value for p in points]
    lines = [p.span for p in points]
    errors = {name: [xs[t] - f(xs[:t]) for t in range(MIN_FIT, len(xs))] for name, f in MENU.items()}
    if len(errors["last value"]) < MIN_ERRORS:
        z = _spread([b - a for a, b in itertools.pairwise(xs)], level)
        return Forecast(xs[-1], xs[-1] - z, xs[-1] + z, "last value; interval from the step volatility", [], lines,
                        anchor=lines[-1])
    name = min(MENU, key=lambda m: st.mean(abs(e) for e in errors[m]))
    point = MENU[name](xs)
    lo, hi = _band(point, errors[name], level)
    return Forecast(point, lo, hi, f"{name}; interval from its {len(errors[name])} backtest errors", errors[name], lines,
                    anchor=lines[-1])


def day(date: str) -> dt.date:
    """A table date as a day: a month (YYYY-MM) as its last day, the end of the period its value is for."""
    if len(date) == 7:
        y, m = int(date[:4]), int(date[5:7])
        return dt.date(y, m, calendar.monthrange(y, m)[1])
    return dt.date.fromisoformat(date[:10])


def weekdays(start: str, end: str) -> int:
    """The weekdays after `start` up to and including `end`."""
    a, b = dt.date.fromisoformat(start[:10]), dt.date.fromisoformat(end[:10])
    return sum((a + dt.timedelta(days=d)).weekday() < 5 for d in range(1, (b - a).days + 1))


def steps_between(points: list[Point], start: str, end: str) -> int:
    """How many observations of the series' own cadence fall after `start` up to `end`: business days for a daily
    series, else the whole periods of the median gap between its dates that end on or before `end`."""
    dates = [day(p.date) for p in points]
    a, b = day(start), day(end)
    gap = st.median((y - x).days for x, y in itertools.pairwise(dates)) if len(dates) > 1 else 1
    if gap <= 3:
        return weekdays(start, end)
    return max(1, math.floor((b - a).days / gap))


def change(points: list[Point], steps: int, scale: float, level: float) -> Forecast:
    xs = [p.value * scale for p in points]
    lines = [p.span for p in points]
    diffs = [b - a for a, b in itertools.pairwise(xs)]
    origins = range(MIN_FIT, len(xs) - steps)
    if len(origins) >= MIN_ERRORS:
        errors = {m: [xs[t + steps] - xs[t] - f(xs[:t + 1], steps) for t in origins] for m, f in CHANGE_MENU.items()}
        name = min(CHANGE_MENU, key=lambda m: st.mean(abs(e) for e in errors[m]))
        point = CHANGE_MENU[name](xs, steps)
        lo, hi = _band(point, errors[name], level)
        return Forecast(point, lo, hi, f"{name} over {steps} steps; interval from its backtest errors", errors[name], lines,
                        anchor=lines[-1])
    half = _spread(diffs, level) * math.sqrt(steps)
    return Forecast(0.0, -half, half, f"no change over {steps} steps; interval from the step volatility of "
                    f"{len(diffs)} steps scaled by the square root of {steps}", diffs, lines, anchor=lines[-1])


def _months(a: str, b: str) -> int:
    return (int(b[:4]) - int(a[:4])) * 12 + int(b[5:7]) - int(a[5:7])


#: A release whose median revision across the reference months is this many times the series' usual mean revision
#: moved the whole history at once (an annual or comprehensive update); the next release is not that kind.
BENCHMARK = 10.0


def _benchmarks(table: dict[str, list[Point]], releases: list[str]) -> set[str]:
    moved = {}
    for prev, d in itertools.pairwise(releases):
        if sizes := [abs(at[d].value - at[prev].value) for est in table.values()
                     if prev in (at := {p.date: p for p in est}) and d in at]:
            moved[d] = sizes
    if len(moved) < 2:
        return set()
    usual = st.median(st.mean(sizes) for sizes in moved.values())
    return {d for d, sizes in moved.items() if st.median(sizes) > BENCHMARK * usual}


def revision(table: dict[str, list[Point]], period: str, latest_vintage: str, latest: float, release: str,
             level: float) -> Forecast:
    releases = sorted({p.date for est in table.values() for p in est})
    own = table.get(period, [])
    if len(releases) < 2:
        return Forecast(latest, latest, latest, "no revision history", [], [p.span for p in own],
                        anchor=own[-1].span if own else None, base=latest)
    # the mean gap counts the releases in a window; a series with two a month (advance and full) has a median gap of
    # about three weeks
    cadence = (dt.date.fromisoformat(releases[-1]) - dt.date.fromisoformat(releases[0])).days / (len(releases) - 1)
    ahead = max(1, round((dt.date.fromisoformat(release) - dt.date.fromisoformat(latest_vintage)).days / cadence))
    age = _months(period, latest_vintage)
    skip = _benchmarks(table, releases)
    same, every, lines = [], [], []
    for ref, est in table.items():
        if ref == period:
            continue
        at = {p.date: p for p in est}
        for i, d in enumerate(releases[:-ahead]):
            later = releases[i + ahead]
            if d in at and later in at and not skip & set(releases[i + 1:i + ahead + 1]):
                r = at[later].value - at[d].value
                every.append(r)
                if _months(ref, d) == age:
                    same.append(r)
                    lines += [at[d].span]
    pool, how = (same, "at the same age") if len(same) >= MIN_FIT or len(every) < 2 else (every, "at any age")
    lines += [p.span for p in own]
    anchor = own[-1].span if own else None
    if not pool:
        return Forecast(latest, latest, latest, "no revision history", [], sorted(set(lines)),
                        anchor=anchor, base=latest)
    point = latest + st.median(pool)
    lo, hi = _band(latest, pool, level)
    lean = point - latest or st.mean(pool)
    label = "up" if lean > 0 else "down" if lean < 0 else None
    return Forecast(point, min(lo, point), max(hi, point),
                    f"latest estimate plus the median of {len(pool)} past {ahead}-release revisions {how}"
                    f"{'' if not skip else ', leaving out the update of ' + ', '.join(sorted(skip))}", pool,
                    sorted(set(lines)), label, anchor=anchor, base=latest)
