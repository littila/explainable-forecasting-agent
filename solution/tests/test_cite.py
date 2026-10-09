import pathlib

from solution import cite, render
from solution.corpus import load_unit

REPO = pathlib.Path(__file__).resolve().parents[2]
COTPOS = load_unit(REPO / "units/t4-cotpos-202411-us10")


def _claim(quote: str, node_id: str = "") -> dict:
    return {"node_id": node_id, "quote": quote, "claim": "x"}


def _lines(unit, docs) -> list[str]:
    return [x.strip() for d in unit.docs if d.doc_id in docs for x in d.text.splitlines() if len(x.split()) >= 8]


def test_a_quote_of_the_entitys_own_document_is_placed_in_its_own_words() -> None:
    own = render.owned(COTPOS, "CORN_CBT")
    line = _lines(COTPOS, own)[0]
    [placed] = cite.place(COTPOS, "CORN_CBT", _claim(line))
    doc = next(d for d in COTPOS.docs if d.doc_id == placed["doc_id"])
    assert doc.doc_id in own and doc.text[placed["span_start"]:placed["span_end"]] == line
    assert placed["claim"] == " ".join(line.split())


def test_another_entitys_document_is_never_cited() -> None:
    own = render.owned(COTPOS, "CORN_CBT")
    mine = set(_lines(COTPOS, own))
    others = {d.doc_id for d in COTPOS.docs} - own
    # a line only the other documents have (a header every document repeats is the entity's own too)
    line = next(x for x in sorted(_lines(COTPOS, others)) if x not in mine)
    # a templated sentence may be placed at the same sentence of the entity's own document, in that document's words
    for placed in cite.place(COTPOS, "CORN_CBT", _claim(line)) or []:
        assert placed["doc_id"] in own and placed["claim"] != " ".join(line.split())
    assert cite.place(COTPOS, None, _claim(line))[0]["doc_id"] in others


def test_a_quote_with_an_ellipsis_resolves_by_its_longest_piece() -> None:
    line = _lines(COTPOS, render.owned(COTPOS, "CORN_CBT"))[0]
    assert cite.place(COTPOS, "CORN_CBT", _claim("made up opening words ... " + line)) is not None


def test_an_invented_quote_is_not_cited() -> None:
    assert cite.place(COTPOS, "CORN_CBT", _claim("the moon is made of cheese on every trading day")) is None


def test_a_filing_quote_resolves_to_its_span_in_the_original_text() -> None:
    unit = load_unit(REPO / "units/t4-postearn-20240201-megacap")
    quote = "Net sales are expected to be between $160.0 billion and $167.0 billion"
    [placed] = cite.place(unit, "AMZN", _claim(quote))
    assert unit.docs[2].text[placed["span_start"]:placed["span_end"]] == quote


def test_zero_width_and_spacing_differences_in_table_rows_still_match() -> None:
    unit = load_unit(REPO / "units/t4-eps-yoy-2023Q2-mixed")
    assert cite.locate(unit, "D1", "Diluted $ ( 0.09 ) $ 0.56") is not None
    assert cite.place(unit, "AMD", _claim("Diluted $ ( 0.09 ) $ 0.56")) is not None


def test_folding_unifies_quotes_dashes_and_invisible_spaces() -> None:
    assert cite.folded("It’s “up” — a\u200b   b")[0] == "it's \"up\" - a b"


def test_a_quote_found_by_its_opening_spans_as_far_as_the_quote_runs_and_ends_on_a_word() -> None:
    unit = load_unit(REPO / "units/t4-credit-event-2023")
    text = unit.docs[0].text
    line = max(text.splitlines(), key=len).strip()
    start = text.index(line)
    opening = " ".join(line[:200].split()[:12])
    quote = opening + " 123,456,789 and other words the passage never had"
    _, at, end = cite.locate(unit, "D1", quote)
    assert at == start and end >= at + len(quote)
    assert end == len(text) or text[end].isspace() or text[end - 1] == "\n"


def test_the_claim_is_the_quote_in_the_passage_s_own_words_else_the_passage() -> None:
    text = "Header\nNet income was $1,234 million — up 5%.\nNext line"
    start, end = 7, text.index("\nNext")
    assert cite.verbatim(text, start, end, "net income was $1,234 million - up 5%") == "Net income was $1,234 million — up 5%"
    assert cite.verbatim(text, start, end, "a quote from elsewhere entirely, eight words long") == "Net income was $1,234 million — up 5%."
    assert cite.verbatim(text, start, end, "Net income") == "Net income was $1,234 million — up 5%."
    assert cite.verbatim(text, start, end, "Net income was $1,234") == "Net income was $1,234"
    long = " ".join(["12,345"] * 200)
    claim = cite.verbatim(long, 0, len(long))
    assert len(claim) <= cite.CLAIM_MAX and claim.split() == ["12,345"] * len(claim.split())


def test_a_claim_quoting_its_own_task_row_cites_each_key_in_the_tables_own_words() -> None:
    unit = load_unit(REPO / "units/t4-fomc-curve-20240918")
    assert cite.place(unit, "UST10Y", _claim("start_yield_pct: 3.73", "task"))[0]["doc_id"] == cite.TASK
    spans = cite.task_spans(unit, "UST10Y", "maturity_years: 10; unit: bps_change")
    assert [unit.task_table[a:b] for a, b in spans] == ['"maturity_years": 10', '"unit": "bps_change"']
    row = unit.task_rows["UST10Y"]
    assert all(row[0] <= a < b <= row[1] for a, b in spans)
    assert [unit.task_table[a:b] for a, b in cite.task_spans(unit, "UST10Y", '"start_yield_pct": 3.73, "as_of": "2024-09-19"')] \
        == ['"start_yield_pct": 3.73', '"as_of": "2024-09-19"']


def test_a_task_quote_with_a_value_the_row_lacks_cites_nothing_and_a_reason_never_cites_the_task() -> None:
    unit = load_unit(REPO / "units/t4-fomc-curve-20240918")
    assert cite.task_spans(unit, "UST10Y", "start_yield_pct: 9.99") == []
    assert cite.task_spans(unit, "UST10Y", "no_such_key: 1") == []
    assert [unit.task_table[a:b] for a, b in cite.task_spans(unit, "UST10Y", "start_yield_pct: 9.99; maturity_years: 10")] \
        == ['"maturity_years": 10']
    assert cite.place(unit, None, _claim("start_yield_pct: 3.73", "task")) is None
