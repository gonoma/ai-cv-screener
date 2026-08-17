import pytest

from backend.domain.query import QueryRouter
from backend.providers.llm_provider import LlmProvider

TABLE = [
    ("Who has experience with Python?", "structured"),
    ("Which candidate graduated from UPC?", "structured"),
    ("How many have 5+ years of experience?", "structured"),
    ("Summarise the profile of Jana Novak.", "profile"),
    ("Who seems strongest at leading teams?", "semantic"),
]


def _stub(monkeypatch, response: dict) -> None:
    monkeypatch.setattr(LlmProvider, "generate_json_object", lambda self, prompt, schema: response)


@pytest.mark.parametrize("question,expected", TABLE)
def test_route_is_passed_through(monkeypatch, question: str, expected: str) -> None:
    _stub(
        monkeypatch,
        {
            "route": expected,
            "name": "Jana Novak" if expected == "profile" else None,
            "skills": [],
            "institution": None,
            "min_years": None,
        },
    )
    assert QueryRouter().classify_question(question).route == expected


def test_unknown_label_degrades_to_semantic(monkeypatch) -> None:
    _stub(
        monkeypatch,
        {"route": "sql", "name": None, "skills": [], "institution": None, "min_years": None},
    )
    assert QueryRouter().classify_question("anything").route == "semantic"


def test_llm_failure_degrades_to_semantic(monkeypatch) -> None:
    def boom(self, prompt: str, _schema: dict) -> dict:
        raise RuntimeError("provider down")

    monkeypatch.setattr(LlmProvider, "generate_json_object", boom)
    assert QueryRouter().classify_question("anything").route == "semantic"


def test_structured_filters_are_carried(monkeypatch) -> None:
    _stub(
        monkeypatch,
        {
            "route": "structured",
            "name": None,
            "skills": ["Python", ""],
            "institution": "UPC",
            "min_years": 5,
        },
    )
    route = QueryRouter().classify_question("who knows Python and studied at UPC with 5+ years")
    assert route.skills == ["Python"]  # empty strings dropped
    assert route.institution == "UPC"
    assert route.minimum_years_experience == 5
