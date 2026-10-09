import pathlib

from solution import plan, render
from solution.corpus import load_unit

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_filing_units_answer_one_entity_a_call_within_the_budget() -> None:
    unit = load_unit(REPO / "units/t4-eps-yoy-2023Q2-mixed")
    calls = plan.calls(unit)
    assert [len(c.ids) for c in calls] == [2, 2, 2]
    budget = int(render.CONTEXT_SHARE * render.WINDOW)
    assert all(render.prompt_tokens(c.system + c.user) <= budget for c in calls)


def test_table_units_split_evenly_within_the_output_plan() -> None:
    unit = load_unit(REPO / "units/t4-cotpos-202411-us10")
    sizes = [len(c.ids) for c in plan.calls(unit)]
    assert sum(sizes) == len(unit.entities)
    assert max(sizes) - min(sizes) <= 1
    assert max(sizes) <= plan.OUTPUT_SHARE * 4_000 // plan.PER_ENTITY


def test_calls_are_named_as_the_lab_names_them() -> None:
    assert [c.name for c in plan.calls(load_unit(REPO / "units/t4-EXAMPLE-eps-beat"))] == [""]
    assert [c.name for c in plan.calls(load_unit(REPO / "units/t4-cpicomp-202410-us11"))] == ["b1-", "b2-"]


def test_every_entity_is_asked_exactly_once_in_roster_order_within_its_call() -> None:
    unit = load_unit(REPO / "units/t4-macrorev-20240930-us6")
    roster = [e.entity_id for e in unit.entities]
    calls = plan.calls(unit)
    assert sorted(e for c in calls for e in c.ids) == sorted(roster)
    assert all(c.ids == sorted(c.ids, key=roster.index) for c in calls)


def test_a_reply_reserve_leaves_fewer_entities_per_call() -> None:
    unit = load_unit(REPO / "units/t4-macrorev-20240930-us6")
    loose = plan.batches(unit, 450, 2_000)
    tight = plan.batches(unit, 450, 2_000, reserve=1_350)
    assert max(len(b.entities) for b in tight) == 4 < max(len(b.entities) for b in loose)
    assert sorted(e for b in tight for e in b.entities) == sorted(e.entity_id for e in unit.entities)


def test_entities_sharing_a_document_of_their_own_are_asked_together() -> None:
    unit = load_unit(REPO / "units/t4-macrorev-20240930-us6")
    tight = plan.batches(unit, 600, 2_000, reserve=1_350)
    series = {e.entity_id: e.entity_id.split("_")[0] for e in unit.entities}
    assert all(len({series[e] for e in b.entities} & {series[x] for x in o.entities}) == 0
               for b in tight for o in tight if o is not b)
    assert max(len(b.entities) for b in tight) == 3
    assert sorted(e for b in tight for e in b.entities) == sorted(series)


def test_ungrouped_batches_are_roster_slices() -> None:
    unit = load_unit(REPO / "units/t4-macrorev-20240930-us6")
    plain = plan.batches(unit, 450, 2_000, reserve=1_350, grouped=False)
    assert [len(b.entities) for b in plain] == [4, 4, 4]
    assert [e for b in plain for e in b.entities] == [e.entity_id for e in unit.entities]


def test_a_group_larger_than_a_call_is_cut_in_roster_order() -> None:
    own = {"a": frozenset({"m", "x"}), "b": frozenset({"m"}), "c": frozenset({"m", "x"}), "d": frozenset({"m", "x"})}
    assert plan.groups(own, 2) == [["a", "c"], ["d"], ["b"]]


def test_a_group_over_the_budget_shares_it_between_its_larger_documents() -> None:
    own = {"a": frozenset({"x", "y", "m"}), "b": frozenset({"z", "m"})}
    size = {"x": 90_000, "y": 90_000, "z": 90_000, "m": 1_000}
    caps = plan.doc_caps([["a"], ["b"]], own, size, 100_000)
    assert caps == {"x": 49_244, "y": 49_244}
    assert sum(min(size[d], caps.get(d, size[d]) + plan.TAG) for d in ("x", "y", "m")) <= 100_000


def test_an_entity_with_two_large_filings_is_planned_not_refused() -> None:
    unit = load_unit(REPO / "units/t4-eps-growth-2024Q3-banks")
    jpm = next(d for d in unit.docs if d.entity_ids == ["JPM"] and len(d.text) > 100_000)
    second = jpm.model_copy(update={"doc_id": "GS_SECOND_FILING", "entity_ids": ["GS"]})
    unit = unit.model_copy(update={"docs": [*unit.docs, second]})
    calls = plan.calls(unit)
    budget = int(render.CONTEXT_SHARE * render.WINDOW)
    assert sorted(e for c in calls for e in c.ids) == sorted(e.entity_id for e in unit.entities)
    assert all(render.prompt_tokens(c.system + c.user) <= budget for c in calls)
    gs = next(c for c in calls if c.ids == ["GS"])
    assert gs.user.count("<doc id=") == 3 and "characters left out" in gs.user
    plain = {c.ids[0]: c.user for c in plan.calls(load_unit(REPO / "units/t4-eps-growth-2024Q3-banks"))}
    assert all(c.user == plain[c.ids[0]] for c in calls if c.ids != ["GS"])
