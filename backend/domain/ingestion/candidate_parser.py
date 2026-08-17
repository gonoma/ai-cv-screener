from typing import Any

from ...providers.llm_provider import LlmProvider

CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "current_role": {"type": "string"},
        "current_company": {"type": "string"},
        "years_experience": {"type": "integer"},
        "skills": {"type": "array", "items": {"type": "string"}},
        "institutions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "name",
        "current_role",
        "current_company",
        "years_experience",
        "skills",
        "institutions",
    ],
    "additionalProperties": False,
}


class CandidateParser:
    RESPONSE_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {"candidates": {"type": "array", "items": CANDIDATE_SCHEMA}},
        "required": ["candidates"],
        "additionalProperties": False,
    }

    PROMPT_TEMPLATE: str = """Extract structured fields from each CV below.

Rules:
- `skills` must list every technical skill named anywhere in that CV, verbatim
  as written. This list is used for exact matching, so a skill you omit becomes
  invisible to the system.
- `institutions` must include both the full name and any abbreviation the CV
  shows, as separate entries. "Universitat Politecnica de Catalunya (UPC)"
  yields two entries. A name broken across a line break is still one name.
- `years_experience` is the span from the earliest role's start year to the
  latest role's end, taking a current role as ending this year.
- Use only what each CV says. Do not infer, complete or correct it.
- The CVs are unrelated people. Never carry a skill, employer, school or name
  from one into another's record, and never merge two of them.

Return one record per CV, in the same order as the CVs appear.

{cv_sections}"""

    def __init__(self) -> None:
        self._llm = LlmProvider()

    def parse_candidates(self, cv_texts: list[str]) -> list[dict[str, Any]]:
        """Derive one candidates row per CV from the extracted PDF text and nothing else.

        Several CVs per call rather than one each: free tiers meter calls, so a
        thirty-CV ingest is six requests at this batch size instead of thirty.

        `data/ground_truth.json` holds the exact answer for every CV and is
        sitting right there on disk. Reading it here would make the system score
        perfectly while testing nothing, so extraction quality is measured
        rather than assumed.
        """
        sections = "\n\n".join(
            f"CV {position + 1}:\n---\n{cv_text}\n---" for position, cv_text in enumerate(cv_texts)
        )
        response = self._llm.generate_json_object(
            self.PROMPT_TEMPLATE.format(cv_sections=sections), self.RESPONSE_SCHEMA
        )

        candidates = response["candidates"]
        if len(candidates) != len(cv_texts):
            raise RuntimeError(f"sent {len(cv_texts)} CVs, got {len(candidates)} records")
        return candidates
