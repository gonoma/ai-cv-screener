from typing import Any

from ...providers.llm_provider import LlmProvider

# One row of a CV's employment history. Dates rather than durations: a model
# asked for "years in this job" has to do arithmetic it is not reliable at,
# while the two years it copies off the page are transcription.
POSITION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "role": {"type": "string"},
        "company": {"type": "string"},
        "start_year": {"type": "integer"},
        # Null is what "still there" looks like, and it has to be expressible:
        # a model forced to name an end year for a current role invents one.
        "end_year": {"type": ["integer", "null"]},
    },
    "required": ["role", "company", "start_year", "end_year"],
    "additionalProperties": False,
}

CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "current_role": {"type": "string"},
        "current_company": {"type": "string"},
        "positions": {"type": "array", "items": POSITION_SCHEMA},
        "skills": {"type": "array", "items": {"type": "string"}},
        "institutions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "name",
        "current_role",
        "current_company",
        "positions",
        "skills",
        "institutions",
    ],
    "additionalProperties": False,
}


class MisalignedBatch(RuntimeError):
    """A batch came back with a different number of records than CVs sent.

    Its own type because the caller can do something about this one: the CVs are
    still on disk and can be asked for singly. A batch that returns one record
    for ten CVs has said nothing about nine of them, and there is no way to tell
    which nine.
    """


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
        - positions: every job the CV lists, in the order it lists them, with the
          years as printed. end_year is null while the role is current ("Present",
          "Now", a dash with nothing after it). Do not compute durations.
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
            raise MisalignedBatch(f"sent {len(cv_texts)} CVs, got {len(candidates)} records")
        return candidates
