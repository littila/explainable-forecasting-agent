import math
import pathlib

import pytest

from solution import extract, forecast, moves
from solution.corpus import load_unit

REPO = pathlib.Path(__file__).resolve().parents[2]
UNIT = load_unit(REPO / "units/t4-fomc-curve-20220728")
QUOTE = "at the 2022-06-16 close the 2-year yield was 3.14 percent and the 10-year 3.28 percent"


def test_an_earlier_level_needs_a_date_before_the_cutoff_and_none_is_an_answer() -> None:
    qualified = moves.starts(UNIT)
    assert qualified["UST10Y"] == ("start_yield_pct", 2.68) and moves.starts(load_unit(REPO / "units/t4-cotpos-202411-us10")) == {}
    kept = f'UST10Y | value: 3.28 | period: 2022-06-16 | quote: "{QUOTE}"'
    late = f'UST10Y | value: 3.28 | period: 2022-08-01 | quote: "{QUOTE}"'
    assert extract.levels(UNIT, kept, qualified, extract.EARLIER_LEVEL)[0]["UST10Y"].value == 3.28
    assert extract.levels(UNIT, late, qualified, extract.EARLIER_LEVEL)[1] == ["UST10Y: the period is not a date before the cutoff"]
    issues = extract.problems(UNIT, qualified, f"{kept}\nUST30Y | none", "stop", ["UST10Y", "UST30Y"],
                              kind=extract.EARLIER_LEVEL)
    assert issues == []


def test_the_least_half_width_scales_the_largest_stated_move_to_the_horizon() -> None:
    level = extract.levels(UNIT, f'UST10Y | value: 3.28 | period: 2022-06-16 | quote: "{QUOTE}"', moves.starts(UNIT),
                           extract.EARLIER_LEVEL)[0]["UST10Y"]
    horizon = forecast.weekdays("2022-07-28", str(UNIT.resolution_date))
    move = 60.0 * math.sqrt(horizon / forecast.weekdays("2022-06-16", "2022-07-28"))
    assert moves.half_width(UNIT, {"UST10Y": level}) == pytest.approx(move)
    assert moves.half_width(UNIT, {}) is None
    assert moves.half_width(load_unit(REPO / "units/t4-cotpos-202411-us10"), {}) is None


def test_an_interval_is_widened_to_the_least_half_width_and_never_narrowed() -> None:
    out = moves.widen({"A": {"point_forecast": 10.0, "int_lo": 0.0, "int_hi": 20.0},
                       "B": {"point_forecast": 0.0, "int_lo": -90.0, "int_hi": 90.0}}, 50.0)
    assert (out["A"]["int_lo"], out["A"]["int_hi"]) == (-40.0, 60.0) and (out["B"]["int_lo"], out["B"]["int_hi"]) == (-90.0, 90.0)


def test_of_several_stated_earlier_levels_the_oldest_is_kept() -> None:
    unit = load_unit(REPO / "units/t4-fomc-curve-20240918")
    quote = "the 2-year yield ended 2024-09-19 at 3.59 percent, roughly 120 basis points below its 4.77 percent close on 2024-07-01"
    text = ('UST2Y | value: 3.61 | period: 2024-09-18 | quote: "2024-09-18 | DGS1=3.95, DGS2=3.61, DGS3=3.49, DGS5=3.47, '
            'DGS7=3.58, DGS10=3.7, DGS20=4.08, DGS30=4.03"\n'
            f'UST2Y | value: 4.77 | period: 2024-07-01 | quote: "{quote}"\nUST10Y | none')
    got, _ = extract.levels(unit, text, moves.starts(unit), extract.EARLIER_LEVEL)
    assert set(got) == {"UST2Y"} and (got["UST2Y"].value, got["UST2Y"].period) == (4.77, "2024-07-01")


@pytest.mark.parametrize(("name", "is_yield_change"), [
    ("yield_change_bps_intermeeting", True), ("yield_move_bp", True), ("ten_year_yield_basis_points", True),
    ("yield_level_pct", False), ("eps_yoy_growth_pct", False)])
def test_a_yield_change_is_a_yield_in_basis_points_by_whole_words(name: str, is_yield_change: bool) -> None:
    assert UNIT.model_copy(update={"target_name": name}).yield_change is is_yield_change
