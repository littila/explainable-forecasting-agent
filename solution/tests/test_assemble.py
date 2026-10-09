import math
import pathlib

import pytest

from solution import cite
from solution.assemble import assemble
from solution.corpus import load_unit
from solution.preflight import preflight
from solution.types import Doc, Entity, Unit

REPO = pathlib.Path(__file__).resolve().parents[2]


def _unit(target_type: str = "regression", labels: list[str] | None = None, n: int = 2) -> Unit:
    return Unit(task_id="t", target_type=target_type, target_name="x", target_labels=labels or [],
                cutoff_date="2024-01-01", entities=[Entity(entity_id=f"E{i}") for i in range(n)],
                docs=[Doc(doc_id="D", doc_date="2024-01-01", text="Revenue grew 5 percent. Margins held.")])


def _assemble(unit: Unit, preds: dict, **kw) -> tuple[dict, list[str]]:
    return assemble(unit, preds, **kw)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "not a number", None])
def test_a_nonfinite_point_becomes_zero(bad: object) -> None:
    answer, _ = _assemble(_unit(), {"E0": {"point_forecast": bad}})
    assert answer["entity_predictions"][0]["point_forecast"] == 0.0


def test_a_label_outside_the_vocabulary_is_coerced_into_it() -> None:
    unit = _unit("classification", ["beat", "miss", "inline"])
    answer, repairs = _assemble(unit, {"E0": {"label": "BEAT!", "point_forecast": 1.0}})
    assert answer["entity_predictions"][0]["label"] == "inline"
    assert any("not in" in r for r in repairs)


def test_an_inverted_interval_is_swapped() -> None:
    answer, repairs = _assemble(_unit(), {"E0": {"point_forecast": 10.0, "interval_lo": 20.0, "interval_hi": 5.0}})
    iv = answer["entity_predictions"][0]["interval"]
    assert (iv["lo"], iv["hi"]) == (5.0, 20.0) and any("inverted" in r for r in repairs)


@pytest.mark.parametrize(("lo", "hi"), [(99.0, 99.0), (1.0, 2.0)])
def test_a_valid_interval_is_kept_as_written_even_zero_width_or_without_its_point(lo: float, hi: float) -> None:
    answer, repairs = _assemble(_unit(), {"E0": {"point_forecast": 99.0, "interval_lo": lo, "interval_hi": hi}})
    iv = answer["entity_predictions"][0]["interval"]
    assert (iv["lo"], iv["hi"]) == (lo, hi) and not [r for r in repairs if r.startswith("E0: interval")]


def test_a_missing_bound_takes_the_point() -> None:
    answer, repairs = _assemble(_unit(), {"E0": {"point_forecast": 7.0, "interval_lo": 5.0}})
    iv = answer["entity_predictions"][0]["interval"]
    assert (iv["lo"], iv["hi"]) == (5.0, 7.0) and any("missing bound" in r for r in repairs)


def test_the_interval_level_is_the_units_not_the_models() -> None:
    unit = _unit()
    answer, _ = _assemble(unit, {})
    assert all(r["interval"]["level"] == unit.interval_level for r in answer["entity_predictions"])


def test_ranking_derives_a_full_permutation_from_point_forecast() -> None:
    unit = _unit("ranking", n=3)
    answer, _ = _assemble(unit, {"E0": {"point_forecast": 1.0}, "E1": {"point_forecast": 3.0},
                                 "E2": {"point_forecast": 2.0}})
    assert [r["rank"] for r in answer["entity_predictions"]] == [3, 1, 2]


def test_assembly_keeps_placed_spans_once_and_drops_what_says_nothing() -> None:
    unit = load_unit(REPO / "units/t4-fomc-curve-20240918")
    e = unit.entities[0].entity_id
    placed = cite.place(unit, e, {"node_id": "", "quote": "2024-09-19 | 3.93 | 3.59 | 3.47 | 3.49 | 3.6 | 3.73 | 4.11 | 4.06"})
    empty = {"doc_id": placed[0]["doc_id"], "span_start": 0, "span_end": 5, "claim": " "}
    preds = {e: {"point_forecast": -5.0, "interval_lo": -20.0, "interval_hi": 10.0, "claims": [*placed, empty, *placed]}}
    answer, repairs = assemble(unit, preds, floor=True)
    first = answer["entity_predictions"][0]["claims"]
    assert first == placed and any("says nothing" in r for r in repairs)
    assert preflight(unit, answer) == []


def test_a_floored_entity_takes_the_label_that_says_nothing_happens() -> None:
    unit = load_unit(REPO / "units/t4-credit-event-2023")
    answer, _ = assemble(unit, {}, floor=True)
    assert {r["label"] for r in answer["entity_predictions"]} == {"no_event"}
    assert {r["point_forecast"] for r in answer["entity_predictions"]} == {0.0}
    claims = [c for r in answer["entity_predictions"] for c in r["claims"]]
    assert len(claims) == len(unit.entities) and all(c["span_end"] - c["span_start"] <= 600 for c in claims)
    assert preflight(unit, answer) == []


def test_without_the_floor_a_missing_citation_stays_visible() -> None:
    answer, repairs = _assemble(_unit(), {})
    assert all(r["claims"] == [] for r in answer["entity_predictions"])
    assert all("NO CITATION" in r for r in repairs if "citation" in r.lower())
    assert math.isfinite(answer["entity_predictions"][0]["interval"]["lo"])


def test_the_content_free_words_are_the_scorers() -> None:
    from qfbench2_track_analysis import scoring

    from solution import assemble as a

    assert a.FUNCTION_WORDS == scoring.CONTENT_FREE_FUNCTION_WORDS
    assert a.META_WORDS == scoring.CONTENT_FREE_META_WORDS
    assert a.FILLER_ANCHORS == scoring.CONTENT_FREE_FILLER_ANCHORS


@pytest.mark.parametrize(("text", "empty"), [
    ("Pre-cutoff evidence selected for the submitted prediction.", True),
    ("Evidence for Example Co from the cited pre-cutoff passage.", True),
    ("Example Co", True),
    ("Guidance was cut.", False),
    ("AAPL has no forecast.", False),
    ("Revenue grew 5 percent.", False),
])
def test_a_claim_that_says_nothing_is_found_as_the_scorer_finds_it(text: str, empty: bool) -> None:
    from solution.assemble import says_nothing

    assert says_nothing(text, ("Example Co", "EX")) is empty


def test_a_claim_citing_more_than_the_span_cap_is_dropped() -> None:
    unit = _unit()
    unit.docs[0].text = "x " * 5_000
    rows = {"E0": {"point_forecast": 1.0, "claims": [{"doc_id": "D", "span_start": 0, "span_end": 9_000, "claim": "x x"}]}}
    answer, repairs = assemble(unit, rows)
    assert answer["entity_predictions"][0]["claims"] == [] and any("over 8000" in r for r in repairs)


def test_an_entity_without_a_forecast_takes_no_change_or_its_rows_latest_value() -> None:
    from solution.assemble import floor_point
    unit = _unit().model_copy(update={"target_name": "next_estimate",
                                      "entities": [Entity(entity_id="E0", features={"latest_fiscal_year": 2024,
                                                                                    "latest_estimate": 5.0})]})
    assert floor_point(unit, unit.entities[0]) == (5.0, "the row's latest_estimate")
    change = unit.model_copy(update={"target_name": "yield_change_bps"})
    assert floor_point(change, change.entities[0]) == (0.0, "no change")
    answer, _ = _assemble(unit, {}, floor=True)
    assert answer["entity_predictions"][0]["point_forecast"] == 5.0


def test_an_entity_with_no_citable_document_cites_its_own_task_row() -> None:
    unit = load_unit(REPO / "units/t4-eps-yoy-2023Q2-mixed")
    amd = next(e for e in unit.entities if e.entity_id == "AMD")
    unit = unit.model_copy(update={"docs": [d for d in unit.docs if "AMD" not in (d.entity_ids or [])]})
    answer, _ = assemble(unit, {}, floor=True)
    row = next(r for r in answer["entity_predictions"] if r["entity_id"] == "AMD")
    [claim] = row["claims"]
    assert claim["doc_id"] == cite.TASK
    assert unit.task_table[claim["span_start"]:claim["span_end"]] == claim["claim"]
    assert any(str(v) in claim["claim"] for v in amd.features.values() if isinstance(v, int | float))
    assert preflight(unit, answer) == []


def test_an_unanswered_entity_takes_the_answered_ones_width_and_label() -> None:
    unit = load_unit(REPO / "units/t4-eps-yoy-2023Q2-mixed")
    ids = [e.entity_id for e in unit.entities]
    claim = {"doc_id": cite.TASK, "span_start": 0, "span_end": 10, "claim": unit.task_table[:10]}
    preds = {e: {"point_forecast": 1.0 + k, "interval_lo": k, "interval_hi": 2.0 + k, "claims": [claim],
                 "label": unit.target_labels[0]} for k, e in enumerate(ids[1:])}
    answer, repairs = assemble(unit, preds, floor=True)
    row = answer["entity_predictions"][0]
    assert row["interval"]["hi"] - row["interval"]["lo"] == 2.0 and row["label"] == unit.target_labels[0]
    assert any("from the 5 answered entities" in r for r in repairs)


@pytest.mark.parametrize("given", ["Up", " up ", '"up"', "**up**", "`UP`"])
def test_a_label_written_with_case_quotes_or_markup_is_the_vocabularys(given: str) -> None:
    unit = _unit().model_copy(update={"target_type": "classification", "target_labels": ["up", "down"]})
    repairs: list[str] = []
    from solution.assemble import _label
    assert _label(unit, given, repairs, "E0") == "up" and repairs == []
    labels = unit.model_copy(update={"target_labels": ["credit_event", "no_event"]})
    assert _label(labels, "No event", repairs, "E0") == "no_event" and _label(labels, "no-event", repairs, "E0") == "no_event"
