import pathlib
import re
from dataclasses import replace

import pytest

from solution import forecast, priors
from solution.corpus import load_unit

REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(("name", "found"), [
    ("t4-auction-btc-202411-us7", 7), ("t4-cotpos-202411-us10", 10), ("t4-cpicomp-202410-us11", 11),
    ("t4-fomc-curve-20240918", 6), ("t4-macrorev-20240930-us6", 12),
    ("t4-fomc-curve-20220728", 0), ("t4-eps-growth-2024Q3-banks", 0), ("t4-credit-event-2023", 0)])
def test_each_entity_with_its_own_dated_history_gets_a_prior_and_no_other(name, found) -> None:
    assert len(priors.priors(load_unit(REPO / "units" / name))) == found


def test_a_revision_prior_is_not_given_to_a_target_that_is_a_change() -> None:
    unit = load_unit(REPO / "units/t4-macrorev-20240930-us6")
    assert len(priors.priors(unit)) == 12
    changed = unit.model_copy(update={"task_id": "macrorev-as-change", "target_name": "next_estimate_revision_change"})
    assert not any(f.decides for f in priors.priors(changed).values())


def test_a_markdown_table_with_outer_pipes_reads_as_a_plain_one() -> None:
    from solution import data
    assert data.cells("| 2024-01 | 0.10 | 0.20 |") == data.cells("2024-01 | 0.10 | 0.20") == ["2024-01", "0.10", "0.20"]
    assert data.cells("|---|---|") == ["---", "---"]


def test_the_series_is_found_by_name_by_target_words_and_as_a_vintage_table() -> None:
    unit = load_unit(REPO / "units/t4-fomc-curve-20240918")
    fomc = priors.priors(unit)["UST10Y"]
    doc, start, end = fomc.anchor
    assert fomc.method.startswith("no change over 34 steps")
    assert next(d.text for d in unit.docs if d.doc_id == doc)[start:end].startswith("2024-09-19 | 3.93")
    unit = load_unit(REPO / "units/t4-auction-btc-202411-us7")
    auction = priors.priors(unit)["AUC_10Y_20241105"]
    doc, start, end = auction.anchor
    assert auction.point == pytest.approx(2.53)
    assert re.match(r"\d{4}-\d{2}-\d{2} \|", next(d.text for d in unit.docs if d.doc_id == doc)[start:end])
    assert auction.anchor == auction.lines[-1]
    revision = priors.priors(load_unit(REPO / "units/t4-macrorev-20240930-us6"))["PAYEMS_2024-07_20241004"]
    assert revision.base == 158637.0 and revision.label == "down"


def _prior(point: float, lo: float, hi: float, base: float | None = None) -> forecast.Forecast:
    return forecast.Forecast(point, lo, hi, "m", [], [("DOC", 0, 9)], anchor=("DOC", 0, 9), base=base)


def test_the_model_gives_the_point_and_the_priors_band_moves_to_it_still_covering_the_prior() -> None:
    unit = load_unit(REPO / "units/t4-auction-btc-202411-us7")
    e = unit.entities[0].entity_id
    out = priors.combine(unit, {e: {"claims": [], "point_forecast": 3.0}}, {e: _prior(2.0, 1.5, 2.5)})[e]
    assert out["point_forecast"] == 3.0 and (out["int_lo"], out["int_hi"]) == (1.5, 3.5)
    assert out["spans"] == [{"doc_id": "DOC", "span_start": 0, "span_end": 9}]


def test_a_prior_of_another_scale_leaves_the_models_forecast_as_it_is() -> None:
    unit = load_unit(REPO / "units/t4-cpicomp-202410-us11")
    index = forecast.Forecast(317.2, 315.9, 318.6, "a level series", [], [])
    block = {"point_forecast": 0.3, "int_lo": 0.1, "int_hi": 0.5, "claims": []}
    out = priors.combine(unit, {"CPI_CORE": block}, {"CPI_CORE": index})["CPI_CORE"]
    assert (out["point_forecast"], out["int_lo"], out["int_hi"]) == (0.3, 0.1, 0.5)


def test_a_ranking_keeps_the_models_point_and_takes_the_priors_width() -> None:
    unit = load_unit(REPO / "units/t4-cotpos-202411-us10")
    e = unit.entities[0].entity_id
    out = priors.combine(unit, {e: {"claims": [], "point_forecast": -3.0}}, {e: _prior(13.0, 10.0, 20.0)})[e]
    assert out["point_forecast"] == -3.0 and (out["int_lo"], out["int_hi"]) == (-8.0, 2.0)


def test_an_entity_the_model_left_out_takes_its_prior_and_a_revision_its_direction() -> None:
    unit = load_unit(REPO / "units/t4-macrorev-20240930-us6")
    e = unit.entities[0].entity_id
    out = priors.combine(unit, {}, {e: _prior(99.0, 95.0, 101.0, base=100.0)})[e]
    assert out["point_forecast"] == 99.0 and out["label"] == "down"


def test_a_revision_takes_the_priors_point_and_direction_whatever_the_model_says() -> None:
    unit = load_unit(REPO / "units/t4-macrorev-20240930-us6")
    e = unit.entities[0].entity_id
    found = priors.priors(unit)[e]
    out = priors.combine(unit, {e: {"claims": [], "point_forecast": found.base + 500.0, "label": "up"}}, {e: found})[e]
    assert out["point_forecast"] == found.point and out["label"] == found.label
    flat = replace(found, point=found.base, label="down")
    assert priors.combine(unit, {e: {"claims": [], "point_forecast": 1.0, "label": "up"}}, {e: flat})[e]["label"] == "down"


def test_an_entity_whose_series_cannot_be_read_leaves_the_others_their_priors(monkeypatch) -> None:
    unit = load_unit(REPO / "units/t4-cotpos-202411-us10")
    every = priors.priors(unit)
    broken = next(iter(every))
    real = priors.prior

    def failing(u, e, rows):
        if e.entity_id == broken:
            raise ValueError("Invalid isoformat string: '2024-09'")
        return real(u, e, rows)

    monkeypatch.setattr(priors, "prior", failing)
    assert set(priors.priors(unit)) == set(every) - {broken}


@pytest.mark.parametrize("name, change", [
    ("yield_change_bps_intermeeting", True), ("eps_yoy_growth_pct", True), ("net_positioning_change_pct_oi_rank", True),
    ("total_return_pct", True), ("exchange_rate", False), ("return_on_equity_pct", False),
    ("cpi_component_mom_first_print", False), ("credit_event_12m", False)])
def test_a_change_target_is_named_by_whole_words(name: str, change: bool) -> None:
    unit = load_unit(REPO / "units/t4-auction-btc-202411-us7")
    assert unit.model_copy(update={"target_name": name}).change_target is change


def test_growth_is_measured_against_the_size_of_the_base_in_the_targets_units() -> None:
    from solution.extract import Level
    unit = load_unit(REPO / "units/t4-eps-growth-2024Q3-banks")
    assert priors.growth(unit, Level(0.83, "Q2", "q", "prior_year_q_eps", 0.90)).point == pytest.approx(-7.7778, abs=1e-4)
    assert priors.growth(unit, Level(-0.5, "Q2", "q", "prior_year_q_eps", -1.0)).point == pytest.approx(50.0)
    ratio = unit.model_copy(update={"target_name": "eps_yoy_growth"})
    assert priors.growth(ratio, Level(1.1, "Q2", "q", "prior_year_q_eps", 1.0)).point == pytest.approx(0.1)


def test_a_bps_target_scales_a_series_only_when_it_is_in_percent() -> None:
    unit = load_unit(REPO / "units/t4-fomc-curve-20240918")
    ten = next(e for e in unit.entities if e.entity_id == "UST10Y")
    assert priors._in_percent(ten, "dgs10", 3.73)
    assert not priors._in_percent(ten, "spread_bps", 125.0)
    assert priors._in_percent(ten, "oas_pct", 1.25)
