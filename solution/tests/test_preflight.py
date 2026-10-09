import copy
import json
import pathlib

from solution.assemble import assemble
from solution.corpus import load_unit
from solution.preflight import VENDORED, preflight
from solution.types import Doc, Entity, Unit

REPO = pathlib.Path(__file__).resolve().parents[2]


def _unit(target_type: str = "regression", labels: list[str] | None = None) -> Unit:
    docs = [Doc(doc_id="D1", doc_date="2024-01-01",
                text="\n\n".join(f"Paragraph {i} runs past the minimum segment length here."
                                 for i in range(6)))]
    return Unit(task_id="t4-x", target_type=target_type, target_name="change",
                target_labels=labels or [], cutoff_date="2024-06-01",
                entities=[Entity(entity_id=f"E{i}", doc_ids=["D1"]) for i in range(3)], docs=docs)


def _clean(unit: Unit) -> dict:
    answer, _ = assemble(unit, {}, floor=True)
    return answer


def test_the_vendored_schema_matches_the_pinned_toolkit() -> None:
    # The whole reason preflight is allowed to run without qfbench2-common installed. If these
    # ever diverge, the image validates against a schema the graders do not use.
    from qfbench2_common.taskcard import schema_path
    packaged = json.loads(pathlib.Path(schema_path("analysis.schema.json")).read_text())
    assert json.loads(VENDORED.read_text()) == packaged


def test_an_assembled_answer_is_clean() -> None:
    unit = _unit()
    assert preflight(unit, _clean(unit)) == []


def test_every_public_unit_assembles_clean() -> None:
    for unit_dir in sorted(p for p in (REPO / "units").iterdir() if p.is_dir()):
        unit = load_unit(unit_dir)
        answer, _ = assemble(unit, {}, floor=True)
        assert preflight(unit, answer) == [], unit_dir.name


def test_a_wrong_task_id_is_caught() -> None:
    unit = _unit()
    answer = _clean(unit) | {"task_id": "t4-something-else"}
    assert any("g2_cutoff_resource" in v for v in preflight(unit, answer))


def test_a_wrong_target_type_is_caught() -> None:
    unit = _unit()
    answer = _clean(unit) | {"target_type": "ranking"}
    assert any("TARGET_TYPE_MISMATCH" in v for v in preflight(unit, answer))


def test_empty_claims_are_caught() -> None:
    unit = _unit()
    answer = _clean(unit)
    answer["entity_predictions"][0]["claims"] = []
    violations = preflight(unit, answer)
    assert any("CITATION_MALFORMED" in v for v in violations)
    assert any("g1_schema" in v for v in violations)


def test_a_missing_entity_is_caught() -> None:
    unit = _unit()
    answer = _clean(unit)
    answer["entity_predictions"].pop()
    assert any("ENTITY_MISSING" in v for v in preflight(unit, answer))


def test_a_duplicate_entity_is_caught() -> None:
    unit = _unit()
    answer = _clean(unit)
    answer["entity_predictions"].append(copy.deepcopy(answer["entity_predictions"][0]))
    assert any("ENTITY_DUPLICATE" in v for v in preflight(unit, answer))


def test_an_unknown_entity_is_caught() -> None:
    unit = _unit()
    answer = _clean(unit)
    answer["entity_predictions"][0]["entity_id"] = "NOT_ON_THE_ROSTER"
    violations = preflight(unit, answer)
    assert any("ENTITY_UNKNOWN" in v for v in violations)


def test_a_wrong_interval_level_is_caught() -> None:
    unit = _unit()
    answer = _clean(unit)
    answer["entity_predictions"][0]["interval"]["level"] = 0.95
    assert any("INTERVAL_LEVEL_MISMATCH" in v for v in preflight(unit, answer))


def test_an_inverted_interval_is_caught() -> None:
    unit = _unit()
    answer = _clean(unit)
    answer["entity_predictions"][0]["interval"] = {"level": 0.9, "lo": 5.0, "hi": -5.0}
    assert any("INTERVAL_INVALID" in v for v in preflight(unit, answer))


def test_a_label_outside_the_vocabulary_is_caught() -> None:
    unit = _unit("classification", labels=["beat", "miss"])
    answer = _clean(unit)
    answer["entity_predictions"][0]["label"] = "sideways"
    assert any("LABEL_INVALID" in v for v in preflight(unit, answer))


def test_a_partial_ranking_is_caught() -> None:
    unit = _unit("ranking")
    answer = _clean(unit)
    del answer["entity_predictions"][0]["rank"]
    assert any("not a ranking" in v for v in preflight(unit, answer))


def test_a_duplicated_rank_is_caught() -> None:
    unit = _unit("ranking")
    answer = _clean(unit)
    answer["entity_predictions"][0]["rank"] = answer["entity_predictions"][1]["rank"]
    assert any("permutation" in v for v in preflight(unit, answer))


def test_a_citation_to_an_unknown_document_is_caught() -> None:
    unit = _unit()
    answer = _clean(unit)
    answer["entity_predictions"][0]["claims"][0]["doc_id"] = "D_DOES_NOT_EXIST"
    assert any("CITATION_UNRESOLVED" in v for v in preflight(unit, answer))


def test_a_post_cutoff_citation_is_caught() -> None:
    unit = _unit()
    unit.docs[0].doc_date = "2025-01-01"   # after the 2024-06-01 cutoff
    assert any("EMBARGO" in v for v in preflight(unit, _clean(unit)))


def test_a_span_past_the_end_of_the_document_is_caught() -> None:
    unit = _unit()
    answer = _clean(unit)
    answer["entity_predictions"][0]["claims"][0]["span_end"] = 10_000
    assert any("SPAN_OUT_OF_RANGE" in v for v in preflight(unit, answer))


def test_a_task_claim_inside_the_entitys_own_row_passes_and_one_outside_it_is_flagged() -> None:
    unit = load_unit(REPO / "units/t4-fomc-curve-20240918")
    answer, _ = assemble(unit, {}, floor=True)
    start, end = unit.task_rows["UST10Y"]
    row = answer["entity_predictions"][[e.entity_id for e in unit.entities].index("UST10Y")]
    row["claims"] = [{"doc_id": "task", "span_start": start + 1, "span_end": end - 1, "claim": "x"}]
    assert preflight(unit, answer) == []
    other = unit.task_rows["UST2Y"]
    row["claims"] = [{"doc_id": "task", "span_start": other[0], "span_end": other[1], "claim": "x"}]
    assert any(v.startswith("TASK_SPAN") for v in preflight(unit, answer))


def test_documents_carry_the_manifest_labels_and_they_decide_what_an_entity_may_cite() -> None:
    from solution import render

    unit = load_unit(REPO / "units/t4-macrorev-20240930-us6")
    files = json.loads((REPO / "units/t4-macrorev-20240930-us6/manifest.json").read_text())["files"]
    marks = {pathlib.Path(f["path"]).stem: (f.get("entity_ids"), bool(f.get("shared"))) for f in files
             if f.get("role") == "corpus"}
    assert {d.doc_id: (d.entity_ids, d.shared) for d in unit.docs} == marks
    for e in unit.entities:
        assert render.owned(unit, e.entity_id) == {d for d, (ids, shared) in marks.items() if shared or e.entity_id in (ids or [])}
