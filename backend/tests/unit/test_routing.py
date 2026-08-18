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

VOCABULARY = (
    ["Jana Novak", "Ada Lovelace"],
    ["Python", "Java", "JavaScript", "Kubernetes"],
    ["Universitat Politecnica de Catalunya", "UPC"],
)


def _stub(monkeypatch, response: dict) -> None:
    monkeypatch.setattr(
        LlmProvider, "generate_json_object", lambda self, prompt, json_schema: response
    )


def _forbid_llm(monkeypatch) -> None:
    """Any call to the provider is a failure, not a slower path.

    These are the questions the corpus is supposed to route on its own; a
    regression here is silent otherwise, because the answer stays correct and
    only the bill changes.
    """

    def spend(self, prompt, json_schema):
        raise AssertionError("routed through the model when the corpus could answer")

    monkeypatch.setattr(LlmProvider, "generate_json_object", spend)


def _router(vocabulary=VOCABULARY) -> QueryRouter:
    built = QueryRouter(connection=None)
    built._vocabulary_cache = vocabulary
    return built


# --- the model still decides when the corpus cannot -------------------------


@pytest.mark.parametrize("question,expected", TABLE)
def test_route_is_passed_through(monkeypatch, question: str, expected: str) -> None:
    _stub(
        monkeypatch=monkeypatch,
        response={
            "route": expected,
            "name": "Jana Novak" if expected == "profile" else None,
            "skills": [],
            "institution": None,
            "min_years": None,
        },
    )
    # No vocabulary, so every question falls through to the model, as it did
    # before the corpus-first router existed.
    assert _router(vocabulary=None).classify_question(question).route == expected


def test_unknown_label_degrades_to_semantic(monkeypatch) -> None:
    _stub(
        monkeypatch=monkeypatch,
        response={
            "route": "sql",
            "name": None,
            "skills": [],
            "institution": None,
            "min_years": None,
        },
    )
    assert _router(vocabulary=None).classify_question("anything").route == "semantic"


def test_llm_failure_degrades_to_semantic(monkeypatch) -> None:
    def boom(self, prompt: str, json_schema: dict) -> dict:
        raise RuntimeError("provider down")

    monkeypatch.setattr(LlmProvider, "generate_json_object", boom)
    assert _router(vocabulary=None).classify_question("anything").route == "semantic"


def test_structured_filters_are_carried(monkeypatch) -> None:
    _stub(
        monkeypatch=monkeypatch,
        response={
            "route": "structured",
            "name": None,
            "skills": ["Python", ""],
            "institution": "UPC",
            "min_years": 5,
        },
    )
    route = _router(vocabulary=None).classify_question("who knows Go and studied somewhere")
    assert route.skills == ["Python"]  # empty strings dropped
    assert route.institution == "UPC"
    assert route.minimum_years_experience == 5


# --- and stays out of it when the corpus can --------------------------------


def test_a_known_name_routes_to_profile_without_spending_anything(monkeypatch) -> None:
    _forbid_llm(monkeypatch)
    route = _router().classify_question("Summarise the profile of Jana Novak.")
    assert route.route == "profile"
    assert route.candidate_name == "Jana Novak"


def test_a_known_skill_with_an_aggregation_cue_routes_to_structured(monkeypatch) -> None:
    _forbid_llm(monkeypatch)
    route = _router().classify_question("Who has experience with Python?")
    assert route.route == "structured"
    assert route.skills == ["Python"]


def test_institution_and_years_are_picked_up_locally(monkeypatch) -> None:
    _forbid_llm(monkeypatch)
    route = _router().classify_question("Which candidates from UPC have 5+ years?")
    assert route.route == "structured"
    assert route.institution == "UPC"
    assert route.minimum_years_experience == 5


def test_a_skill_is_not_matched_inside_a_longer_one(monkeypatch) -> None:
    """The failure that makes a cheap router worse than an expensive one.

    Substring matching would route a JavaScript question into a filter for
    Java and return a confident, wrong list of people — which is worse than the
    round trip it saved.
    """
    _forbid_llm(monkeypatch)
    assert _router().classify_question("who knows JavaScript?").skills == ["JavaScript"]


def test_an_open_question_still_reaches_the_model(monkeypatch) -> None:
    """An aggregation cue alone must not become a filterless structured query.

    "who" is in this question and nothing else is; routing it structurally
    would return the entire corpus as context and answer a question about
    judgement with a directory listing.
    """
    _stub(
        monkeypatch=monkeypatch,
        response={
            "route": "semantic",
            "name": None,
            "skills": [],
            "institution": None,
            "min_years": None,
        },
    )
    assert _router().classify_question("Who would suit a startup?").route == "semantic"


def test_no_connection_and_no_cache_falls_back_to_the_model(monkeypatch) -> None:
    _stub(
        monkeypatch=monkeypatch,
        response={
            "route": "semantic",
            "name": None,
            "skills": [],
            "institution": None,
            "min_years": None,
        },
    )
    assert QueryRouter().classify_question("who knows Python?").route == "semantic"


# --- forcing a route ---------------------------------------------------------


def test_a_forced_route_still_gets_its_filters_for_free(monkeypatch) -> None:
    """Forcing `structured` with no filters means SELECT * — the costliest prompt possible."""
    _forbid_llm(monkeypatch)
    route = _router().force_route(
        route_name="structured",
        question="who knows Kubernetes?",
    )
    assert route.route == "structured"
    assert route.skills == ["Kubernetes"]


def test_a_forced_route_overrides_what_the_corpus_would_have_picked(monkeypatch) -> None:
    _forbid_llm(monkeypatch)
    route = _router().force_route(
        route_name="semantic",
        question="who knows Kubernetes?",
    )
    assert route.route == "semantic"
    assert route.skills == []
