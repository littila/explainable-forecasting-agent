import pathlib

from solution import prompt
from solution.corpus import load_unit

REPO = pathlib.Path(__file__).resolve().parents[2]


def _unit(name: str):
    return load_unit(REPO / "units" / name)


def test_one_quote_rule_for_every_unit_allows_a_whole_table_row() -> None:
    for name in ("t4-cotpos-202411-us10", "t4-postearn-20240201-megacap", "t4-macrorev-20240930-us6"):
        text = prompt.system(_unit(name))
        assert prompt.QUOTE in text and "[D" not in text.split("Sample (")[1]


def test_the_system_text_asks_for_a_reason_per_entity_and_no_reasoning_line() -> None:
    text = prompt.system(_unit("t4-cotpos-202411-us10"))
    assert "<reasons> holds one reason for each entity" in text and "at least 3 in all" in text
    assert "\nreasoning:" not in text
    assert text.split("Sample (")[1].count('<reason n="') == 4


def test_the_user_text_runs_documents_task_entities_then_the_closing_line() -> None:
    unit = _unit("t4-auction-btc-202411-us7")
    ids = [e.entity_id for e in unit.entities]
    text = prompt.user(unit, ids[:2], None)
    order = [text.index(x) for x in ("<documents>", "<task>", "<entities>", "Answer for these entities")]
    assert order == sorted(order)
    assert f"in this order: {ids[0]}, {ids[1]}. Reply with" in text
    assert ids[2] not in text.split("<entities>")[1]


def test_a_labelled_unit_shows_its_labels_in_the_format_and_the_sample() -> None:
    text = prompt.system(_unit("t4-macrorev-20240930-us6"))
    assert "\nlabel: one of up, down." in text
    assert "\nlabel: up\npoint_forecast: 126" in text


def test_each_document_carries_its_manifest_label() -> None:
    unit = _unit("t4-cotpos-202411-us10")
    text = prompt.user(unit, [unit.entities[0].entity_id], None)
    assert 'about="market"' in text
    assert f'about="{unit.entities[0].entity_id}"' in text


def test_an_entity_without_a_name_takes_its_name_feature() -> None:
    unit = _unit("t4-macrorev-20240930-us6")
    first = prompt.entity_lines(unit.entities).splitlines()[0]
    assert first.startswith(f"{unit.entities[0].entity_id} — ")


def test_the_task_block_states_the_target_label_definitions_and_dates() -> None:
    unit = load_unit(REPO / "units/t4-postearn-20240201-megacap")
    text = prompt.user(unit, [e.entity_id for e in unit.entities], frozenset())
    task = text[text.index("<task>"):text.index("</task>")]
    assert "target: earnings_reaction (classification)" in task
    assert all(f"label {k}: {v}" in task for k, v in unit.label_assertions.items())
    assert f"cutoff date: {unit.cutoff_date}; resolution date: {unit.resolution_date}" in task
    assert "; the label is" not in prompt.system(unit)
