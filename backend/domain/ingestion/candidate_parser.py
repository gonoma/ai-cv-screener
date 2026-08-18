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

    # Every word here is resent with every batch, so the rules are kept to the
    # ones that change the output. The field list is not repeated in prose — the
    # JSON schema already carries it, and stating it twice pays twice.
    PROMPT_TEMPLATE: str = """
        Extract fields from each CV. One record per CV, same order.

        - skills: every technical skill named anywhere in that CV, verbatim. Used for
          exact matching, so an omitted skill becomes invisible.
        - institutions: full name and any abbreviation as separate entries. A name
          broken across a line break is still one name.
        - years_experience: earliest role's start year to the latest role's end, a
          current role ending this year.
        - Only what the CV says. Do not infer or correct.
        - Unrelated people: never move a skill, employer, school or name between
          records, never merge two.
        
        {cv_sections}
    """

    def __init__(self) -> None:
        self._llm = LlmProvider()

    def parse_candidates(self, cv_texts: list[str]) -> list[dict[str, Any]]:
        """Derive one candidates row per CV from the extracted PDF text and nothing else.

        Several CVs per call because the rules above are the same for all of
        them: one batch of ten pays for the instructions once instead of ten
        times. That is a smaller win than it sounds — the CV bodies dominate the
        prompt, so batching buys a few percent, not an order of magnitude — and
        it is only free because a batch that comes back misaligned is now
        re-asked one CV at a time rather than re-sending the whole batch.

        `data/ground_truth.json` holds the exact answer for every CV and is
        sitting right there on disk. Reading it here would make the system score
        perfectly while testing nothing, so extraction quality is measured
        rather than assumed.
        """
        sections = "\n\n".join(
            f"CV {position + 1}:\n---\n{cv_text}\n---" for position, cv_text in enumerate(cv_texts)
        )
        response = self._llm.generate_json_object(
            prompt=self.PROMPT_TEMPLATE.format(cv_sections=sections),
            json_schema=self.RESPONSE_SCHEMA,
        )

        candidates = response["candidates"]
        if len(candidates) != len(cv_texts):
            raise RuntimeError(f"sent {len(cv_texts)} CVs, got {len(candidates)} records")
        return candidates
