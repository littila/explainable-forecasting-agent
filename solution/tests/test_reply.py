import pathlib

from solution import reply
from solution.corpus import load_unit

REPO = pathlib.Path(__file__).resolve().parents[2]
UNIT = load_unit(REPO / "units/t4-fomc-curve-20240918")
EVERY = [e.entity_id for e in UNIT.entities]
ROWS = [{"entity_id": e, "point_forecast": 3.6, "interval": {"level": 0.9, "lo": 3.4, "hi": 3.8}, "claims": []}
        for e in EVERY]
CLOSE = '"2024-09-19 | 3.93 | 3.59 | 3.47 | 3.49 | 3.6 | 3.73 | 4.11 | 4.06"'
EVE = '"2024-09-18 | 3.95 | 3.61 | 3.49 | 3.47 | 3.58 | 3.7 | 4.08 | 4.03"'
PATH = '"Recent indicators suggest that economic activity has continued to expand at a solid pace."'


def _reason(n: int, citation: str, premise: str, scope: str = "") -> str:
    return (f'<reason n="{n}">\ncitation: {citation}\npremise: {premise}\nmechanism: m{n}.\nimplication: i{n}.\n'
            + (f"scope: {scope}\n" if scope else "") + "</reason>")


def _reply(reasons: list[str], ids: list[str]) -> str:
    blocks = [f'<entity id="{e}">\nclaim: {CLOSE}\npoint_forecast: 3.6\nint_lo: 3.4\nint_hi: 3.8\n</entity>' for e in ids]
    return "<reasons>\n" + "\n".join(reasons) + "\n</reasons>\n<forecasts>\n" + "\n".join(blocks) + "\n</forecasts>"


def test_a_reply_reads_into_reasons_in_their_place_and_entity_blocks() -> None:
    reasons, blocks = reply.parse(_reply([_reason(1, CLOSE, "p1"), _reason(2, PATH, "p2", "UST2Y")], ["UST2Y"]))
    assert [(r["premise"], r["_position"], r.get("scope")) for r in reasons] == [("p1", 1, None), ("p2", 2, ["UST2Y"])]
    assert blocks["UST2Y"]["point_forecast"] == 3.6
    assert blocks["UST2Y"]["claims"][0]["quote"] == CLOSE.strip('"')


def test_each_calls_reasons_carry_the_call_and_its_entities() -> None:
    reasons, blocks = reply.parse_calls([(["UST2Y"], [_reply([_reason(1, CLOSE, "p")], ["UST2Y"])]),
                                         (["UST10Y"], [_reply([_reason(1, PATH, "q")], ["UST10Y"])])])
    assert [(r["_call"], r["_ids"]) for r in reasons] == [(1, ["UST2Y"]), (2, ["UST10Y"])]
    assert sorted(blocks) == ["UST10Y", "UST2Y"]


def test_a_corrected_reply_missing_an_entity_keeps_that_entity_from_the_reply_it_corrects() -> None:
    first = _reply([_reason(1, CLOSE, "first")], ["UST2Y", "UST10Y"])
    second = _reply([_reason(1, PATH, "second")], ["UST2Y"]).replace("3.6", "3.7")
    reasons, blocks = reply.parse_calls([(["UST2Y", "UST10Y"], [second, first])])
    assert [r["premise"] for r in reasons] == ["second"]
    assert blocks["UST2Y"]["point_forecast"] == 3.7 and blocks["UST10Y"]["point_forecast"] == 3.6


def test_a_later_call_fills_only_entities_without_a_forecast() -> None:
    answered = _reply([_reason(1, CLOSE, "p")], ["UST2Y"])
    again = _reply([_reason(1, PATH, "q")], ["UST2Y"]).replace("3.6", "9.9")
    _, blocks = reply.parse_calls([(["UST2Y"], [answered]), (["UST2Y"], [again])])
    assert blocks["UST2Y"]["point_forecast"] == 3.6


def test_selection_keeps_the_three_reasons_about_most_of_the_roster() -> None:
    text = _reply([_reason(1, CLOSE, "about one", "UST2Y"), _reason(2, PATH, "about all"),
                   _reason(3, EVE, "about two", "UST2Y, UST5Y"), _reason(4, CLOSE, "about three", "UST2Y, UST5Y, UST7Y")],
                  EVERY)
    kept, dropped = reply.selected(UNIT, reply.parse_calls([(EVERY, [text])])[0], ROWS)
    assert [r["premise"] for r in kept] == ["about all", "about three", "about two"]
    assert [r["reason_id"] for r in kept] == ["r1", "r2", "r3"]
    assert dropped == ["reason 1: beyond the first 3"]


def test_selection_refuses_a_deny_listed_reason_and_another_calls_reason_on_a_chosen_passage() -> None:
    first = _reply([_reason(1, CLOSE, "Yields closed lower"), _reason(2, PATH, "see the leaderboard")], ["UST2Y"])
    second = _reply([_reason(1, CLOSE, "Closing yields fell"), _reason(2, EVE, "Yields a day earlier")], ["UST10Y"])
    reasons = reply.parse_calls([(["UST2Y"], [first]), (["UST10Y"], [second])])[0]
    kept, dropped = reply.selected(UNIT, reasons, ROWS)
    assert [r["premise"] for r in kept] == ["Yields closed lower", "Yields a day earlier"]
    assert "reason 2: refused (deny_list)" in dropped
    assert "reason 3: cites a passage a reason of another call already chosen cites" in dropped


def test_a_passage_two_filings_share_is_cited_from_the_one_the_reason_is_about() -> None:
    unit = load_unit(REPO / "units/t4-eps-yoy-2023Q2-mixed")
    amgen = next(d for d in unit.docs if d.doc_id.endswith("0318154_10Q_20230428")).entity_ids[0]
    quote = reply.quoted('"In the first quarter of 2023, we initiated a workforce rebalancing action"')
    rows = [{"entity_id": e.entity_id, "point_forecast": 1.0, "interval": {"level": 0.9, "lo": 0.0, "hi": 2.0},
             "claims": []} for e in unit.entities]
    reasons = [{"premise": "p", "mechanism": "m", "implication": "i", "citations": [quote], "scope": [amgen],
                "_call": 1, "_position": 1, "_ids": [e.entity_id for e in unit.entities]}]
    kept, _ = reply.selected(unit, reasons, rows)
    assert kept[0]["citations"][0]["doc_id"].endswith("0318154_10Q_20230428")


def test_a_reason_with_citations_ranks_before_a_broader_one_without() -> None:
    text = _reply([_reason(1, '"no such passage anywhere in the corpus at all"', "uncited, about all"),
                   _reason(2, CLOSE, "cited, about one", "UST2Y")], EVERY)
    kept, _ = reply.selected(UNIT, reply.parse_calls([(EVERY, [text])])[0], ROWS)
    assert [r["premise"] for r in kept] == ["cited, about one", "uncited, about all"]


def test_a_quote_followed_by_a_note_or_an_odd_bracket_is_still_read() -> None:
    assert reply.quoted('"a b" [D3.12] and surrounding context') == {"node_id": "D3.12", "quote": "a b", "claim": "a b"}
    assert reply.quoted('"a b" [D7.10-11, approx lines]')["node_id"] == ""
    assert reply.quoted('"he said "yes" today" [D1.2]')["quote"] == 'he said "yes" today'


def test_a_block_with_a_label_and_no_number_keeps_its_label_and_numbers_read_in_e_notation() -> None:
    text = ('<forecasts>\n<entity id="A">\nlabel: Up\n</entity>\n<entity id="B">\npoint_forecast: 1.2e-3\n'
            "int_lo: -2E-3\nint_hi: 5e-3\n</entity>\n</forecasts>")
    _, blocks = reply.parse(text)
    assert blocks["A"]["label"] == "Up" and blocks["A"].get("point_forecast") is None
    assert (blocks["B"]["point_forecast"], blocks["B"]["int_lo"], blocks["B"]["int_hi"]) == (0.0012, -0.002, 0.005)
    unit = UNIT.model_copy(update={"entities": [e.model_copy(update={"entity_id": x}) for e, x in zip(UNIT.entities, "AB")]})
    assert set(reply.predictions(unit, blocks)) == {"A", "B"}


def test_an_entity_id_with_a_space_is_read_quoted_and_an_unquoted_one_still_is() -> None:
    from solution.reply import parse
    text = ('<forecasts>\n<entity id="S&P 500">\npoint_forecast: 1.5\nint_lo: 1\nint_hi: 2\n</entity>\n'
            '<entity id=EURUSD>\npoint_forecast: 2\nint_lo: 1\nint_hi: 3\n</entity>\n'
            '<entity id="SILVER_CMX>\npoint_forecast: 3\nint_lo: 1\nint_hi: 4\n</entity>\n</forecasts>')
    assert {e: b["point_forecast"] for e, b in parse(text)[1].items()} == {"S&P 500": 1.5, "EURUSD": 2.0,
                                                                         "SILVER_CMX": 3.0}
