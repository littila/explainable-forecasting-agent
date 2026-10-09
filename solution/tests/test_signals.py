import pathlib

from solution import signals
from solution.corpus import load_unit

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_going_concern_doubt_and_an_investment_grade_rating_decide_credit_labels() -> None:
    assert signals.labels(load_unit(REPO / "units/t4-credit-event-2023")) == {
        "BBBY": "credit_event", "BBY": "no_event", "WBA": "no_event", "WE": "credit_event"}
    assert signals.labels(load_unit(REPO / "units/t4-eps-yoy-2023Q2-mixed")) == {}


def test_doubt_must_be_concluded_and_a_split_rating_is_not_investment_grade() -> None:
    assert signals.doubt("the Company has concluded that there is substantial doubt about the Company's ability to "
                         "continue as a going concern")
    assert not signals.doubt("Management evaluated whether there are conditions and events that raise substantial "
                             "doubt about the Company's ability to continue as a going concern within one year.")
    assert not signals.doubt("did not raise substantial doubt about our ability to continue as a going concern")
    assert signals.investment_grade("Our credit ratings were: Moody's Baa3, Standard & Poor's BBB.")
    assert not signals.investment_grade("Our credit rating: Moody's Ba1, Standard & Poor's BB+, Fitch BBB-.")


def test_a_decided_label_mirrors_a_probability_on_the_other_side() -> None:
    blocks = {"WE": {"label": "no_event", "point_forecast": 0.3, "int_lo": 0.1, "int_hi": 0.4},
              "BBY": {"label": "no_event", "point_forecast": 0.05}}
    out, changed = signals.apply(blocks, {"WE": "credit_event", "BBY": "no_event"})
    assert changed == ["WE"] and out["BBY"] == blocks["BBY"]
    assert (out["WE"]["label"], out["WE"]["point_forecast"], out["WE"]["int_lo"], out["WE"]["int_hi"]) == (
        "credit_event", 0.7, 0.6, 0.9)
