import re
from typing import Any

import psycopg

from ...data.models import ROUTES, QueryRoute
from ...providers.llm_provider import LlmProvider


class QueryRouter:
    # Words that mean "every candidate matching a condition" rather than "the
    # ones most like this". Only consulted alongside a term the corpus actually
    # contains, so "who is good with people" does not become a structured query
    # with no filters — that would return the whole table.
    AGGREGATION_CUES: tuple[str, ...] = (
        "who",
        "which",
        "how many",
        "list",
        "all",
        "anyone",
        "everyone",
        "count",
    )

    # "5+ years", "at least 7 years", "more than 3 years of experience"
    YEARS_PATTERN = re.compile(r"(\d{1,2})\s*\+?\s*(?:or more\s*)?year", re.IGNORECASE)

    RESPONSE_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "route": {"type": "string", "enum": list(ROUTES)},
            "name": {"type": ["string", "null"]},
            "skills": {"type": "array", "items": {"type": "string"}},
            "institution": {"type": ["string", "null"]},
            "min_years": {"type": ["integer", "null"]},
        },
        "required": ["route", "name", "skills", "institution", "min_years"],
        "additionalProperties": False,
    }

    # Trimmed to the distinctions the model actually gets wrong. The examples
    # that survive are the ones that separate structured from semantic, because
    # that is the boundary where a mistake costs a wrong answer rather than a
    # slower one.
    PROMPT_TEMPLATE: str = """
    Classify a recruiter's question about a CV corpus.

    structured — an exact filter or count over all candidates, where the right
      answer is *every* match. "who knows Kubernetes", "how many have 5+ years".
    profile — scoped to one named person; put the name in `name`.
    semantic — qualitative, no exact field answers it. "who suits a startup".
    
    For structured only, fill `skills` (exact technologies named), `institution`,
    `min_years`. Otherwise leave them empty or null.
    
    Question: {question}
    """

    def __init__(self, connection: psycopg.Connection | None = None) -> None:
        self._llm = LlmProvider()
        self._connection = connection

    def classify_question(self, question: str) -> QueryRoute:
        """Route the question, reaching for the model only when the corpus cannot decide."""
        routed_locally = self._route_from_corpus_vocabulary(question)
        if routed_locally is not None:
            return routed_locally
        return self._route_with_model(question)

    def force_route(self, route_name: str, question: str) -> QueryRoute:
        derived = self._route_from_corpus_vocabulary(question)
        if derived is None or derived.route != route_name:
            return QueryRoute(route=route_name)
        return derived

    def _route_with_model(self, question: str) -> QueryRoute:
        try:
            response = self._llm.generate_json_object(
                prompt=self.PROMPT_TEMPLATE.format(question=question),
                json_schema=self.RESPONSE_SCHEMA,
            )
        except Exception:
            return QueryRoute(route="semantic")

        route = str(response.get("route", "")).strip().lower()
        if route not in ROUTES:
            return QueryRoute(route="semantic")

        return QueryRoute(
            route=route,
            candidate_name=response.get("name") or None,
            skills=[skill for skill in (response.get("skills") or []) if skill],
            institution=response.get("institution") or None,
            minimum_years_experience=response.get("min_years"),
        )

    def _route_from_corpus_vocabulary(self, question: str) -> QueryRoute | None:
        """Answer the routing question from what is already in the database, or return None."""
        vocabulary = self._corpus_vocabulary()
        if vocabulary is None:
            return None

        names, skills, institutions = vocabulary
        lowered_question = question.lower()

        matched_name = self._longest_match(lowered_question, names)
        if matched_name:
            return QueryRoute(route="profile", candidate_name=matched_name)

        if not any(cue in lowered_question for cue in self.AGGREGATION_CUES):
            return None

        matched_skills = [skill for skill in skills if self._contains_term(lowered_question, skill)]
        matched_institution = self._longest_match(lowered_question, institutions)
        years = self.YEARS_PATTERN.search(question)

        if not matched_skills and not matched_institution and not years:
            return None

        return QueryRoute(
            route="structured",
            skills=matched_skills,
            institution=matched_institution,
            minimum_years_experience=int(years.group(1)) if years else None,
        )

    def _corpus_vocabulary(self) -> tuple[list[str], list[str], list[str]] | None:
        """Every name, skill and institution in the corpus, in one query.

        Cached for the life of the router, which is one request: the corpus only
        changes on ingest, and re-reading it per question would trade tokens for
        round trips to Postgres.
        """
        if not hasattr(self, "_vocabulary_cache"):
            if self._connection is None:
                return None
            try:
                names, skills, institutions = self._connection.execute(
                    "SELECT (SELECT array_agg(DISTINCT name) FROM candidates), "
                    "(SELECT array_agg(DISTINCT s) FROM candidates, unnest(skills) s), "
                    "(SELECT array_agg(DISTINCT i) FROM candidates, unnest(institutions) i)"
                ).fetchone()
            except Exception:
                self._vocabulary_cache = None
            else:
                self._vocabulary_cache = (names or [], skills or [], institutions or [])
        return self._vocabulary_cache

    @classmethod
    def _longest_match(cls, lowered_question: str, terms: list[str]) -> str | None:
        """Longest first, so "Ana Silva Costa" never resolves to "Ana Silva"."""
        for term in sorted(terms, key=len, reverse=True):
            if cls._contains_term(lowered_question, term):
                return term
        return None

    @staticmethod
    def _contains_term(lowered_question: str, term: str) -> bool:
        """Whole-term match, so "Java" does not match inside "JavaScript"."""
        if not term.strip():
            return False
        return re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", lowered_question) is not None
