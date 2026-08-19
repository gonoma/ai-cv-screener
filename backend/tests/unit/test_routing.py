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


# --- superlatives, which are an ordering rather than a filter ---------------


def test_the_longest_stint_in_one_job_is_a_ranking(monkeypatch) -> None:
    """Names no skill and no school, so it used to fall through to plain retrieval."""
    _forbid_llm(monkeypatch)

    route = _router().classify_question(
        "Which candidate has worked for the longest period in a single job or position?"
    )

    assert (route.route, route.ranking) == ("structured", "tenure")
    assert route.skills == []


def test_the_longest_experience_in_a_skill_is_a_ranking_and_a_filter(monkeypatch) -> None:
    _forbid_llm(monkeypatch)

    route = _router().classify_question(
        "Which candidate has the longest experience in Python and why?"
    )

    assert (route.route, route.ranking) == ("structured", "experience")
    assert route.skills == ["Python"]


def test_a_threshold_question_is_still_a_filter_not_a_ranking(monkeypatch) -> None:
    """ "5+ years" selects rows; it does not ask which row is highest."""
    _forbid_llm(monkeypatch)

    route = _router().classify_question("Who has 5+ years of experience with Java?")

    assert route.ranking is None
    assert route.minimum_years_experience == 5


def test_a_superlative_does_not_double_as_a_threshold(monkeypatch) -> None:
    """A ranking that also filtered on the number in the question would rank a subset."""
    _forbid_llm(monkeypatch)

    route = _router().classify_question("Who has the most years of experience beyond 3 years?")

    assert route.ranking == "experience"
    assert route.minimum_years_experience is None


def test_a_qualitative_superlative_is_not_a_ranking(monkeypatch) -> None:
    """ "Strongest at leading teams" is not a column, so it must stay semantic."""
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

    assert (
        _router().classify_question("Who is the most impressive communicator?").route == "semantic"
    )


# --- a term used as an illustration is not a filter -------------------------


def test_a_skill_named_only_in_an_example_is_not_filtered_on(monkeypatch) -> None:
    """Filtering on it answered a question about thirty people with sixteen of them."""
    _forbid_llm(monkeypatch)

    route = _router().classify_question(
        "Can you divide all 30 candidates by roughly what they do? For example: 33% do "
        "frontend, 40% do backend with Python, 2% machine learning."
    )

    assert (route.route, route.skills, route.breakdown) == ("structured", [], True)


def test_a_name_named_only_in_an_example_does_not_become_a_profile(monkeypatch) -> None:
    """One CV cited as an illustration must not narrow the question to that person."""
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

    route = _router().classify_question(
        "How many candidates are backend engineers, e.g. Jana Novak?"
    )

    assert route.route != "profile"
    assert route.candidate_name is None


def test_a_filter_stated_before_the_example_survives(monkeypatch) -> None:
    """Cutting at the marker must not cut the question itself."""
    _forbid_llm(monkeypatch)

    route = _router().classify_question(
        "Who knows Python, such as people who have shipped Kubernetes work?"
    )

    assert route.skills == ["Python"]


def test_a_breakdown_of_a_filtered_group_keeps_the_filter(monkeypatch) -> None:
    _forbid_llm(monkeypatch)

    route = _router().classify_question("Break down the Python candidates by seniority")

    assert (route.skills, route.breakdown) == (["Python"], True)
