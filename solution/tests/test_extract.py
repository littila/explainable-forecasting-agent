import pathlib

import pytest

from solution import extract, priors
from solution.corpus import load_unit

REPO = pathlib.Path(__file__).resolve().parents[2]
UNIT = load_unit(REPO / "units/t4-eps-growth-2024Q3-banks")
QUOTE = "Diluted earnings per common share $ 0.83"


def test_only_a_growth_target_with_a_base_in_its_row_qualifies() -> None:
    assert extract.bases(UNIT, set())["BAC"] == ("prior_year_q_eps", 0.9)
    assert extract.bases(load_unit(REPO / "units/t4-eps-yoy-2023Q2-mixed"), set()) == {}
    assert extract.bases(load_unit(REPO / "units/t4-fomc-curve-20220728"), set()) == {}


def test_a_level_is_kept_only_when_its_quote_and_number_check_out() -> None:
    qualified = {"BAC": ("prior_year_q_eps", 0.9)}
    good = f'BAC | value: 0.83 | period: three months ended June 30, 2024 | quote: "{QUOTE}"'
    got, dropped = extract.levels(UNIT, good, qualified)
    assert got["BAC"].value == 0.83 and dropped == []
    wrong_number = f'BAC | value: 0.85 | period: Q2 2024 | quote: "{QUOTE}"'
    invented = 'BAC | value: 0.83 | period: Q2 2024 | quote: "Diluted earnings were a record 0.83 per share this quarter"'
    assert extract.levels(UNIT, wrong_number, qualified)[0] == {}
    assert extract.levels(UNIT, invented, qualified)[0] == {}



def test_a_leading_dot_decimal_in_the_quote_is_read_as_a_decimal() -> None:
    qualified = {"USB": ("prior_year_q_eps", 1.05)}
    line = ('USB | value: 0.97 | period: three months ended June 30, 2024 | '
            'quote: "Diluted earnings per share .97 .84 15.5 1.75 1.87 (6.4)"')
    got, dropped = extract.levels(UNIT, line, qualified)
    assert got["USB"].value == 0.97 and dropped == []

def test_a_level_becomes_a_growth_prior_that_takes_the_point_and_stretches_the_models_band() -> None:
    got, _ = extract.levels(UNIT, f'BAC | value: 0.83 | period: Q2 2024 | quote: "{QUOTE}"', {"BAC": ("prior_year_q_eps", 0.9)})
    prior = priors.growth(UNIT, got["BAC"])
    assert prior.point == pytest.approx((0.83 / 0.9 - 1) * 100)
    out = priors.combine(UNIT, {"BAC": {"claims": [], "point_forecast": 2.0, "int_lo": 0.0, "int_hi": 4.0}}, {"BAC": prior})["BAC"]
    assert out["point_forecast"] == pytest.approx(prior.point)
    shift = prior.point - 2.0
    assert out["int_lo"] == pytest.approx(0.0 + shift) and out["int_hi"] == pytest.approx(2.0)
    assert out["claims"][-1]["quote"] == QUOTE


def test_a_copy_of_the_base_from_a_row_of_periods_takes_the_rows_first_column() -> None:
    qualified = extract.bases(UNIT, set())
    line = ('JPM | value: 4.33 | period: three months ended September 30, 2023 | '
            'quote: "Diluted 6.12 4.44 3.04 4.33 4.75 10.56 8.85"')
    got, dropped = extract.levels(UNIT, line, {"JPM": qualified["JPM"]})
    assert qualified["JPM"][1] == 4.33 and got["JPM"].value == 6.12 and dropped == []


def test_dollar_signs_inserted_into_a_quote_are_folded_away() -> None:
    qualified = extract.bases(UNIT, set())
    line = ('USB | value: 0.97 | period: three months ended June 30, 2024 | '
            'quote: "Diluted earnings per share $ .97 $ .84 15.5 $ 1.75 $ 1.87 (6.4)"')
    got, _ = extract.levels(UNIT, line, {"USB": qualified["USB"]})
    assert got["USB"].value == 0.97 and "$" not in got["USB"].quote


def test_a_period_after_the_filings_date_is_named_as_the_reason() -> None:
    qualified = extract.bases(UNIT, set())
    line = ('GS | value: 10.01 | period: Three Months Ended September 2024 | '
            'quote: "Diluted EPS $ 10.01 $ 5.47 $ 20.21 $ 11.91"')
    got, dropped = extract.levels(UNIT, line, {"GS": qualified["GS"]})
    assert got == {} and "ends after the date of its documents" in dropped[0]


def test_a_base_named_as_a_rate_does_not_make_a_growth_prior() -> None:
    unit = load_unit(REPO / "units/t4-eps-growth-2024Q3-banks")
    rows = [e.model_copy(update={"features": {"prior_quarter_growth": 3.2}}) for e in unit.entities]
    assert extract.bases(unit.model_copy(update={"entities": rows}), set()) == {}


def test_a_copied_level_far_from_its_base_is_left_out() -> None:
    unit = load_unit(REPO / "units/t4-eps-growth-2024Q3-banks")
    qualified = {"BAC": ("prior_year_q_eps", 0.90)}
    line = 'BAC | value: 6,975 | period: Three Months Ended June 30 2024 | quote: "Net income 6,975 7,410"'
    got, dropped = extract.levels(unit, line, qualified)
    assert got == {} and dropped == ["BAC: the value is not between a fifth and five times the base; copy the same quantity"]
