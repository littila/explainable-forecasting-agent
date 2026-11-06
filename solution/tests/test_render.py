import itertools
import pathlib
import re

from solution import render
from solution.corpus import load_unit

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_every_document_is_shown_as_its_original_text_in_its_doc_tag() -> None:
    unit = load_unit(REPO / "units/t4-fomc-curve-20240918")
    shown = render.rendered(unit)
    assert shown.count("<doc id=") == len(unit.docs) and "[D1." not in shown
    assert all(d.text.strip() in shown for d in unit.docs)
    one = unit.docs[1].doc_id
    assert render.rendered(unit, frozenset({one})).startswith('<doc id="D2"')


def test_an_unlabelled_document_is_taken_as_market_wide() -> None:
    unit = load_unit(REPO / "units/t4-fomc-curve-20240918")
    assert render.owned(unit, "UST10Y") == frozenset(d.doc_id for d in unit.docs)


def test_fit_keeps_text_that_fits_and_collapses_whitespace_before_cutting() -> None:
    short = "Revenue rose.   Costs fell."
    assert render.fit(short, "revenue", 10_000) is short
    spaced = "word   \n " * 3_000
    assert render.fit(spaced, "word", 10_000) == re.sub(r"\s+", " ", spaced).strip()


def test_fit_keeps_the_best_windows_in_document_order_with_gaps_marked(monkeypatch) -> None:
    monkeypatch.setattr(render, "WINDOW_CHARS", 1_000)
    filler = "The weather in the region was mild throughout the period under review. " * 40
    target = "The company disclosed substantial doubt about its ability to continue as a going concern. " * 12
    text = filler + target + filler
    assert render.prompt_tokens(text) > 600
    out = render.fit(text, "going concern substantial doubt", 600)
    assert "going concern" in out
    assert "characters left out" in out and len(out) < len(text)
    assert render.prompt_tokens(out) <= 600


def test_a_cap_smaller_than_one_window_still_keeps_the_best_passage() -> None:
    filler = "The weather in the region was mild throughout the period under review. " * 400
    target = "The company disclosed substantial doubt about its ability to continue as a going concern. "
    text = filler + target + filler
    assert len(text) > 3 * render.WINDOW_CHARS
    out = render.fit(text, "going concern substantial doubt", 400)
    assert "going concern" in out and render.prompt_tokens(out) <= 400


def test_windows_cover_the_text_without_gaps_or_overlaps(monkeypatch) -> None:
    monkeypatch.setattr(render, "WINDOW_CHARS", 500)
    text = "A sentence of moderate length ends here. " * 60
    spans = render.windows(text)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(a[1] == b[0] for a, b in itertools.pairwise(spans))
    assert all(text[b - 2:b] == ". " for _, b in spans[:-1])


def _planted(tmp_path: pathlib.Path, name: str, doc_glob: str, old: str, new: str) -> pathlib.Path:
    """A copy of a unit (under its own task id, so no cache is shared) with `old` replaced by `new` in one document."""
    import json
    import shutil
    dst = tmp_path / name
    shutil.copytree(REPO / "units" / name, dst)
    task = json.loads((dst / "task.json").read_text())
    task["task_id"] = f"{task['task_id']}-planted"
    (dst / "task.json").write_text(json.dumps(task))
    path = next(dst.glob(f"corpus/{doc_glob}"))
    doc = json.loads(path.read_text())
    assert old in doc["text"]
    doc["text"] = doc["text"].replace(old, new)
    path.write_text(json.dumps(doc))
    return dst


def test_a_row_dated_after_the_cutoff_in_an_admitted_document_is_shown_and_read_as_history(tmp_path) -> None:
    from solution import data
    name = "t4-fomc-curve-20240918"
    row = "2024-10-15 | 9.99 | 9.99 | 9.99 | 9.99 | 9.99 | 9.99 | 9.99 | 9.99\n"
    planted = load_unit(_planted(tmp_path, name, "RATES_SNAPSHOT_*.json", "2024-09-19 | 3.93", row + "2024-09-19 | 3.93"))
    clean = load_unit(REPO / "units" / name)
    assert row.strip() in render.rendered(planted)
    doc = next(d.doc_id for d in clean.docs if "RATES" in d.doc_id)
    values = lambda u: [(p.date, p.value) for p in data.series(u, doc, "dgs10")]
    assert values(planted) == [*values(clean), ("2024-10-15", 9.99)] and values(clean)


def test_a_release_column_dated_after_the_cutoff_in_an_admitted_document_is_shown_and_read(tmp_path) -> None:
    import json

    from solution import data
    name = "t4-macrorev-20240930-us6"
    dst = _planted(tmp_path, name, "ALFRED_PAYEMS_*.json", "| as_of_2024-09-06\n", "| as_of_2024-09-06 | as_of_2024-10-04\n")
    path = next(dst.glob("corpus/ALFRED_PAYEMS_*.json"))
    doc = json.loads(path.read_text())
    doc["text"] = re.sub(r"^(2024-\d{2} \|.*)$", r"\1 | 999999", doc["text"], flags=re.MULTILINE)
    path.write_text(json.dumps(doc))
    planted, clean = load_unit(dst), load_unit(REPO / "units" / name)
    doc_id = next(d.doc_id for d in planted.docs if "PAYEMS" in d.doc_id)
    assert "999999" in render.rendered(planted) and "as_of_2024-10-04" in render.rendered(planted)
    values = lambda u: {k: [(p.date, p.value) for p in v] for k, v in data.vintages(u, doc_id).items()}
    later = {k: v for k, v in values(planted).items() if k.startswith("2024-")}
    assert later and all(v == [*values(clean)[k], ("2024-10-04", 999999.0)] for k, v in later.items())
