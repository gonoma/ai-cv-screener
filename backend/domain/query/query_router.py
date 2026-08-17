from typing import Any

from ...data.models import ROUTES, QueryRoute
from ...providers.llm_provider import LlmProvider


class QueryRouter:
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

    PROMPT_TEMPLATE: str = """Classify a recruiter's question about a corpus of CVs
into exactly one route.

structured — an aggregation or an exact filter over all candidates. "Who has
  experience with Python", "who studied at UPC", "how many have 5+ years",
  "who knows both Kubernetes and Terraform". Choose this whenever the correct
  answer is *every* candidate matching a condition, because a similarity search
  would silently return only the first few.
profile — scoped to one named person. "Summarise Jane Doe", "what does Marc
  Oliver do". Put the name in `name`.
semantic — open-ended or qualitative, where no exact field answers it. "Who
  seems strongest at leading teams", "who would suit a startup".

Also fill, for the structured route only:
- `skills`: the exact technologies named in the question, [] if none
- `institution`: the school or university named, null if none
- `min_years`: a minimum years-of-experience threshold, null if none

Leave them empty or null for the other routes.

Question: {question}"""

    def __init__(self) -> None:
        self._llm = LlmProvider()

    def classify_question(self, question: str) -> QueryRoute:
        """Pick one of three routes, falling back to `semantic` on any surprise.

        The bare except is deliberate rather than careless: an unreachable
        provider, malformed JSON or an unknown label should degrade to ordinary
        RAG, which still answers, instead of failing the user's request. The
        filters come back on this same call so the structured route needs one
        round trip rather than classify-then-derive.
        """
        try:
            response = self._llm.generate_json_object(
                self.PROMPT_TEMPLATE.format(question=question), self.RESPONSE_SCHEMA
            )
        # A routing failure must not fail the request: semantic is ordinary RAG
        # and an acceptable floor.
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
