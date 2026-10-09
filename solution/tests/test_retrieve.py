import pathlib

from solution.corpus import load_unit
from solution.retrieve import BM25, retrieval_query, task_terms

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_a_text_carrying_the_query_terms_outranks_one_that_does_not() -> None:
    texts = ["The company disclosed substantial doubt about its ability to continue as a going concern.",
             "Our corporate headquarters occupy leased office space in three metropolitan regions."]
    assert BM25(texts).ranked("going concern substantial doubt") == [0]


def test_ranking_is_deterministic_under_ties() -> None:
    texts = [f"Identical wording about revenue, instance {i}." for i in range(5)]
    assert BM25(texts).ranked("revenue") == [0, 1, 2, 3, 4]


def test_a_stopword_only_query_matches_nothing() -> None:
    assert BM25(["The revenue grew.", "Margins fell."]).ranked("the of and") == []


def test_task_terms_keep_the_target_and_its_drivers_and_drop_the_instructions() -> None:
    unit = load_unit(REPO / "units/t4-eps-growth-2024Q3-banks")
    terms = task_terms(unit).split()
    assert "diluted" in terms and "interest" in terms and "provisions" in terms
    assert not {"frozen", "corpus", "cutoff", "predict", "citations"} & set(terms)


def test_the_query_names_the_target_not_an_entity() -> None:
    unit = load_unit(REPO / "units/t4-eps-growth-2024Q3-banks")
    query = retrieval_query(unit).lower()
    assert "eps" in query and "growth" in query
    assert not any(e.name.lower() in query for e in unit.entities)
