"""A unit's answer calls (lab v4.12's planner, ported 2026-10-04).

The entities are split into calls, entities that share a document of their own (one not every entity has) in the
same call. A call's prompt stays within the render's call budget (`render.CONTEXT_SHARE` of `render.WINDOW`): the
prompt text, the documents every entity shares, and the batch's own documents. Its reply plans on PER_ENTITY output
tokens an entity (its block and its own reason) and SHARED reasons besides, within OUTPUT_SHARE of the platform's
output cap, and at most BATCH_MAX entities. The plan is the
fewest calls that hold every group: groups placed largest first, in roster order among equals, each in the first call
with room, the calls first given even shares of the roster; a call's entities, and the calls, keep roster order. A
group larger than a call is cut in roster order. A group whose documents together exceed the budget (an entity with two
large filings, a filing about two entities) has its larger documents fitted to a smaller cap, the largest at which the
group fits (`doc_caps`); the render fits each document on its own to `render.DOC_MAX`, which holds one large filing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from solution import prompt, render
from solution.llm import MAX_OUTPUT
from solution.types import Unit

#: A reply plans on this share of the output cap, so a call has room for a longer reply.
OUTPUT_SHARE = 0.85
BATCH_MAX = 16
#: Reply tokens a reason and an entity block plan on: lab v4.11's means (242 and 135) with about 20% added. An entity
#: plans on its block and its own reason; a call on SHARED reasons besides (6 entities a call; 3 reasons on one).
PER_REASON = 300
PER_BLOCK = 160
PER_ENTITY = PER_BLOCK + PER_REASON
SHARED = 2
#: Tokens a fitted document may take beyond its cap: its <doc> tag and gap markers.
TAG = 256


@dataclass(frozen=True)
class Batch:
    entities: list[str]
    docs: frozenset[str]
    #: The documents fitted below `render.DOC_MAX`, with their caps.
    caps: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Call:
    name: str
    ids: list[str]
    system: str
    user: str


def groups(own: dict[str, frozenset[str]], most: int) -> list[list[str]]:
    """The entities in `own` (roster order) grouped by the documents they share beyond those every entity has, in
    roster order of their first entity, each cut to at most `most` entities."""
    common = frozenset.intersection(*own.values())
    order = {e: k for k, e in enumerate(own)}
    out: list[list[str]] = []
    for e, docs in own.items():
        linked = [g for g in out if any(own[x] & docs - common for x in g)]
        out = [g for g in out if g not in linked] + [sorted([x for g in linked for x in g] + [e], key=order.get)]
    out.sort(key=lambda g: order[g[0]])
    return [g[k:k + most] for g in out for k in range(0, len(g), most)]


def doc_caps(parts: list[list[str]], own: dict[str, frozenset[str]], size: dict[str, int], budget: int) -> dict[str, int]:
    """For each group whose documents together exceed `budget` tokens, the largest cap that fits them, given to each of
    its documents larger than that cap (the smallest cap where a document is in several such groups)."""
    out: dict[str, int] = {}
    for g in parts:
        docs = sorted(frozenset().union(*(own[e] for e in g)))
        if sum(size[d] for d in docs) <= budget:
            continue
        lo, hi = 0, render.DOC_MAX
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if sum(min(size[d], mid + TAG) for d in docs) <= budget else (lo, mid - 1)
        for d in docs:
            if size[d] > lo + TAG:
                out[d] = min(out.get(d, lo), lo)
    return out


def packed(parts: list[list[str]], caps: list[int], own: dict[str, frozenset[str]], size: dict[str, int],
           budget: int) -> list[list[str]] | None:
    """`parts` placed largest first (stable among equals) in the first of len(caps) calls with room: at most caps[k]
    entities, their documents within `budget` tokens; None when one does not fit."""
    out: list[list[str]] = [[] for _ in caps]
    for g in sorted(parts, key=len, reverse=True):
        for b, cap in zip(out, caps):
            if len(b) + len(g) <= cap and sum(size[d] for d in frozenset().union(*(own[e] for e in b + g))) <= budget:
                b += g
                break
        else:
            return None
    return out


def bins(parts: list[list[str]], own: dict[str, frozenset[str]], size: dict[str, int], budget: int,
         most: int) -> list[list[str]]:
    """The fewest calls that hold every group (see the module docstring), each call's entities in roster order."""
    order = {e: k for k, e in enumerate(own)}
    n = len(own)
    for k in range(math.ceil(n / most), len(parts) + 1):
        for caps in ([n // k + (j < n % k) for j in range(k)], [most] * k):
            if out := packed(parts, caps, own, size, budget):
                return sorted((sorted(b, key=order.get) for b in out if b), key=lambda b: order[b[0]])
    raise AssertionError("a call for each group always fits")


def calls_at(cap: int, parts: list[list[str]], own: dict[str, frozenset[str]], size: dict[str, int], budget: int,
             most: int, capped: dict[str, int]) -> int:
    """How many calls the groups take with every document fitted to at most `cap` tokens (counting TAG over it)."""
    est = {d: min(s, min(capped.get(d, cap), cap) + TAG) for d, s in size.items()}
    return len(bins(parts, own, est, budget, most))


def fewer(parts: list[list[str]], own: dict[str, frozenset[str]], size: dict[str, int], budget: int, most: int,
          most_calls: int, capped: dict[str, int]) -> dict[str, int]:
    """`capped` with every document lowered to the largest cap at which the groups pack into `most_calls` calls; as
    given when no cap gets there (more entities than `most_calls` calls hold)."""
    if calls_at(0, parts, own, size, budget, most, capped) > most_calls:
        return capped
    lo, hi = 0, render.DOC_MAX
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if calls_at(mid, parts, own, size, budget, most, capped) <= most_calls else (lo, mid - 1)
    return capped | {d: min(capped.get(d, lo), lo) for d, s in size.items() if s > lo + TAG}


def batches(unit: Unit, per_entity: int, overhead: int, reserve: int = 0, *, grouped: bool = True,
            most_calls: int | None = None) -> list[Batch]:
    """The unit's entities split into calls (see the module docstring); `overhead` is the prompt's tokens without
    documents, `reserve` the reply tokens every call spends besides its entities (a reply's reasons). `grouped=False`
    places each entity on its own (lab v4.8). With `most_calls`, documents are fitted smaller until the calls are no
    more than that, as far as the output plan allows. Raises when a group's documents exceed the budget even fitted."""
    own = {e.entity_id: render.owned(unit, e.entity_id) for e in unit.entities}
    size = {d.doc_id: render.prompt_tokens(render.rendered(unit, frozenset({d.doc_id}))) for d in unit.docs}
    budget = int(render.CONTEXT_SHARE * render.WINDOW) - overhead
    most = min(BATCH_MAX, max(1, (int(OUTPUT_SHARE * MAX_OUTPUT) - reserve) // per_entity))
    parts = groups(own, most) if grouped else [[e] for e in own]
    capped = doc_caps(parts, own, size, budget)
    if most_calls is not None:
        est = {d: min(s, capped[d] + TAG) if d in capped else s for d, s in size.items()}
        if len(bins(parts, own, est, budget, most)) > most_calls:
            capped = fewer(parts, own, size, budget, most, most_calls, capped)
    size |= {d: render.prompt_tokens(render.rendered(unit, frozenset({d}), capped)) for d in capped}
    for g in parts:
        cost = sum(size[d] for d in frozenset().union(*(own[e] for e in g)))
        if cost > budget:
            raise ValueError(f"{', '.join(g)}: their documents take {cost:,} tokens, over the {budget:,} a call may use")
    return [Batch(b, docs := frozenset().union(*(own[e] for e in b)), {d: capped[d] for d in docs & capped.keys()})
            for b in bins(parts, own, size, budget, most)]


def calls(unit: Unit, decided: dict[str, str] | None = None, most_calls: int | None = None) -> list[Call]:
    """The unit's answer calls, named as the lab names them ("" for a single call, else "b1-", "b2-", ...), with the
    forecasts code decides shown on their entities' lines (`decided`), at most `most_calls` of them where the output
    plan allows."""
    system = prompt.system(unit)
    every = [e.entity_id for e in unit.entities]
    overhead = render.prompt_tokens(system + prompt.user(unit, every, frozenset(), decided))
    plan = batches(unit, PER_ENTITY, overhead, reserve=SHARED * PER_REASON, most_calls=most_calls)
    if len(plan) == 1:
        return [Call("", plan[0].entities, system, prompt.user(unit, plan[0].entities, None, decided, plan[0].caps))]
    return [Call(f"b{k}-", b.entities, system, prompt.user(unit, b.entities, b.docs, decided, b.caps))
            for k, b in enumerate(plan, 1)]
