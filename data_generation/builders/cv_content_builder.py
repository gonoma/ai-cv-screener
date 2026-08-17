import os
from typing import Any

from ..models.candidate_specs import CandidateSpec
from ..models.cv_records import CORPUS_YEAR, Candidate, CandidateBatch


class CvContentBuilder:
    _GEMINI_MODEL_NAME: str = "gemini-3.7-flash"
    _OPENROUTER_MODEL_NAME: str = "nvidia/nemotron-3-super-120b-a12b:free"
    _OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    _DEFAULT_PROVIDER: str = "gemini"

    _DIVERSITY_BRIEF: str = f"""You generate synthetic CV data for testing a \
    CV-screening system. Every profile is fictional. The corpus is dated \
    {CORPUS_YEAR}: no role may end after {CORPUS_YEAR}, and the current role has a \
    null end_year.
    
    The single most important property of this corpus is variety. Left alone, \
    generated profiles converge on one archetype: a mid-level full-stack engineer \
    in a large city with a computer science degree and the same eight skills. That \
    corpus would test nothing. Push deliberately away from the archetype on every \
    axis below.
    
    - Seniority: interns and juniors through staff, principal and director. Not \
    everyone is senior.
    - Function: backend, frontend, mobile, data engineering, ML, SRE/platform, QA, \
    security, product management, UX design, technical writing, engineering \
    management.
    - Career shape: linear climbers, career changers arriving from teaching or \
    finance, long tenures at one employer, contractors with many short posts, \
    people returning after a gap, part-time and freelance work.
    - Geography: mostly Spain and the rest of Europe, some outside it. Vary the \
    city, not just the country. Do not put the whole corpus in one city.
    - Names: span the origins a European tech hiring pool actually contains — \
    Castilian, Catalan, Basque, Galician, Portuguese, Italian, French, German, \
    Polish, Romanian, Maghrebi, South Asian, Chinese, Latin American. Vary gender \
    too. Surnames from a single region across the corpus is the failure mode to \
    avoid.
    - Presentation and appearance: fill both fields so the photograph matches the \
    person rather than contradicting them. `presentation` must agree with the given \
    name. Vary it across the corpus — not everyone is a woman, and a non-binary \
    candidate should appear occasionally rather than never.
    - `appearance` is what stops thirty headshots reading as one casting call, so \
    spread it deliberately. Ages from the early twenties to the late forties, \
    tracking career length. Builds from slim through average to heavy, with heavy \
    appearing as often as slim. Ordinary, unremarkable faces rather than striking \
    ones: these are people photographed for a work profile, not for a magazine, and \
    a corpus of uniformly attractive candidates is the failure mode. Give most of \
    them one plain, specific detail — glasses, a receding hairline, freckles, \
    crooked teeth, a double chin, thinning hair, acne scarring, a strong nose, a \
    gap tooth. Skin tone and hair should follow from the name's origin without \
    being predictable from it.
    - Education: Spanish and European universities (UPC, UPM, UB, UAB, Universidad \
    de Sevilla, TU Delft, KTH, Politecnico di Milano, Trinity College Dublin and \
    others), plus bootcamp graduates, self-taught engineers with no degree at all, \
    and the occasional PhD. Not every candidate has a university education.
    - Skills: real, specific, and consistent with the role. A QA engineer's list \
    should not read like a backend engineer's. Some candidates should overlap \
    heavily on common technologies and some barely overlap with anyone.
    - Languages: vary the number and the levels.
    
    Write plausible achievement bullets with concrete numbers where a real CV would \
    have them. Do not use emoji. Return only the record."""

    def build_candidates(self, specs: list[CandidateSpec]) -> list[Candidate]:
        """
        Produce one candidate per spec in a single call.
        Costs one call per batch instead of one per CV.
        """
        prompt = self._DIVERSITY_BRIEF + self._build_spec_section(specs)
        provider = os.environ.get("TEXT_PROVIDER", self._DEFAULT_PROVIDER).strip().lower()
        if provider == "gemini":
            batch = self._request_batch_from_gemini(prompt)
        elif provider == "openrouter":
            batch = self._request_batch_from_openrouter(prompt)
        else:
            raise SystemExit(f"TEXT_PROVIDER must be 'gemini' or 'openrouter', got {provider!r}")

        if len(batch.candidates) != len(specs):
            raise RuntimeError(f"asked for {len(specs)} records, got {len(batch.candidates)}")
        return batch.candidates

    @staticmethod
    def _build_spec_section(specs: list[CandidateSpec]) -> str:
        briefs = [f"{position + 1}. {spec.as_brief()}" for position, spec in enumerate(specs)]
        return (
            f"\n\nWrite {len(specs)} candidate records, one for each brief below, in this "
            "order. Treat each brief as fixed: the role, seniority, city, years of "
            "experience, education, industry, career shape and first language are given, "
            "and everything else — name, employers, skills, achievements, appearance — is yours to "
            "invent so that the people read as unrelated to one another.\n\n" + "\n".join(briefs)
        )

    def _request_batch_from_gemini(self, prompt: str) -> CandidateBatch:
        self._required_key("GEMINI_API_KEY")
        from google import genai

        # Bound to a name, not chained: `Client().interactions` does not keep its
        # parent alive, so a one-liner closes the httpx pool before the request.
        client = genai.Client()
        interaction = client.interactions.create(
            model=self._GEMINI_MODEL_NAME,
            input=prompt,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": CandidateBatch.model_json_schema(),
            },
        )
        return CandidateBatch.model_validate_json(interaction.output_text)

    def _request_batch_from_openrouter(self, prompt: str) -> CandidateBatch:
        import openai

        client = openai.OpenAI(
            base_url=self._OPENROUTER_BASE_URL,
            api_key=self._required_key("OPENROUTER_API_KEY"),
        )
        completion = client.chat.completions.create(
            model=self._OPENROUTER_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "candidate_batch",
                    "strict": True,
                    "schema": self._forbid_extra_properties(CandidateBatch.model_json_schema()),
                },
            },
        )
        return CandidateBatch.model_validate_json(completion.choices[0].message.content)

    @staticmethod
    def _required_key(variable_name: str) -> str:
        api_key = os.environ.get(variable_name)
        if not api_key:
            raise SystemExit(f"{variable_name} is not set. Add it to .env.")
        return api_key

    @classmethod
    def _forbid_extra_properties(cls, schema: dict[str, Any]) -> dict[str, Any]:
        """Walk the schema adding `additionalProperties: false` to every object.

        Pydantic does not emit it and OpenRouter's strict mode rejects any
        object that omits it, nested models included — so this is required for
        the request to be accepted at all, not a tightening.
        """
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
        for member in schema.values():
            if isinstance(member, dict):
                cls._forbid_extra_properties(member)
            elif isinstance(member, list):
                for entry in member:
                    if isinstance(entry, dict):
                        cls._forbid_extra_properties(entry)
        return schema
