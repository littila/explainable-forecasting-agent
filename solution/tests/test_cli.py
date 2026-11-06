import json
import pathlib

from solution import cli, pipeline

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_a_valid_answer_is_written_before_the_pipeline_and_replaced_after_it(monkeypatch, tmp_path) -> None:
    out = tmp_path / "answer.json"
    seen = {}

    def failing(unit, *, deadline_at=None, floor=False):
        seen["first"] = json.loads(out.read_text())
        raise RuntimeError("the model is gone")

    monkeypatch.setattr(pipeline, "build_answer", failing)
    unit = REPO / "units/t4-macrorev-20240930-us6"
    assert cli.run(unit / "task.json", unit / "corpus", out) == 0
    assert seen["first"]["notes"]["failed"] == "written before any model call"
    final = json.loads(out.read_text())
    assert len(final["entity_predictions"]) == 12 and all(r["claims"] for r in final["entity_predictions"])
    from solution import priors
    from solution.corpus import load_unit
    found = priors.priors(load_unit(unit))
    first = final["entity_predictions"][0]
    assert first["point_forecast"] == found[first["entity_id"]].point
    assert not list(tmp_path.glob("*.tmp"))


def test_a_unit_that_does_not_load_still_gets_an_answer_the_schema_admits(monkeypatch, tmp_path) -> None:
    from solution import corpus
    from solution.corpus import load_unit
    from solution.preflight import preflight

    def broken(*args, **kwargs):
        raise ValueError("an unexpected task shape")

    out = tmp_path / "answer.json"
    unit = REPO / "units/t4-eps-yoy-2023Q2-mixed"
    monkeypatch.setattr(corpus, "load_unit", broken)
    assert cli.run(unit / "task.json", unit / "corpus", out) == 0
    answer = json.loads(out.read_text())
    assert answer["notes"]["agent"] == "solution.cli last-resort"
    monkeypatch.undo()
    loaded = load_unit(unit)
    assert all(r["claims"] and r["label"] in loaded.target_labels for r in answer["entity_predictions"])
    assert preflight(loaded, answer) == []


def test_a_unit_in_shapes_the_practice_units_do_not_use_still_loads(tmp_path) -> None:
    import shutil

    from solution.corpus import load_unit

    src = REPO / "units/t4-eps-yoy-2023Q2-mixed"
    unit_dir = tmp_path / "unit"
    shutil.copytree(src, unit_dir)
    task = json.loads((unit_dir / "task.json").read_text())
    labels = task["target"].pop("labels")
    task["resolution_date"] = task["resolution_date"] + "T00:00:00Z"
    task["entities"][0]["name"] = None
    task["entities"][1]["corpus_ref"] = "/input/corpus/"
    task["entities"][2]["corpus_ref"] = "./corpus/"
    (unit_dir / "task.json").write_text(json.dumps(task))
    first = min((unit_dir / "corpus").glob("*.json"))
    doc = json.loads(first.read_text())
    words = doc.pop("text").split(" ")
    doc["spans"] = [{"text": " ".join(words[:50])}, {"text": " ".join(words[50:])}]
    doc["doc_id"] = "SOMETHING_ELSE"
    first.write_text(json.dumps(doc))
    unit = load_unit(unit_dir)
    assert unit.target_labels == labels and unit.resolution_date == task["resolution_date"][:10]
    assert {d.doc_id for d in unit.docs} == {p.stem for p in (unit_dir / "corpus").glob("*.json") if p.stem != "manifest"}
    assert next(d for d in unit.docs if d.doc_id == first.stem).text == " ".join(words)
    assert unit.entities[0].name == "" and all(e.doc_ids for e in unit.entities)


def test_a_null_label_loads_and_a_file_the_manifest_does_not_declare_is_left_out(tmp_path) -> None:
    import shutil

    from solution.corpus import load_unit

    unit_dir = tmp_path / "unit"
    shutil.copytree(REPO / "units/t4-eps-yoy-2023Q2-mixed", unit_dir)
    manifest = json.loads((unit_dir / "manifest.json").read_text())
    entry = next(f for f in manifest["files"] if f.get("role") == "corpus")
    entry["entity_ids"] = None
    (unit_dir / "manifest.json").write_text(json.dumps(manifest))
    first = min((unit_dir / "corpus").glob("*.json"))
    (unit_dir / "corpus" / "UNDECLARED_DOC.json").write_text(first.read_text())
    unit = load_unit(unit_dir)
    ids = {d.doc_id for d in unit.docs}
    assert "UNDECLARED_DOC" not in ids and pathlib.Path(entry["path"]).stem in ids


def test_the_deadline_follows_the_cards_time_limit(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("T4_HARD_DEADLINE_SEC", raising=False)
    unit = REPO / "units/t4-fomc-curve-20240918"
    assert cli.hard_deadline(unit / "task.json") == 570.0
    (tmp_path / "card.toml").write_text("[agent]\ntimeout_sec = 120.0\n")
    assert cli.hard_deadline(tmp_path / "task.json") == 108.0
    (tmp_path / "card.toml").write_text("[agent]\ntimeout_sec = 1800\n")
    assert cli.hard_deadline(tmp_path / "task.json") == 1770.0
    assert cli.hard_deadline(tmp_path / "nowhere" / "task.json") == cli.HARD_DEADLINE
