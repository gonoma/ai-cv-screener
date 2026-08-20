"""Marks the answers: does this answer match what the answer key says is true?

Everything here is plain Python — set comparisons, regular expressions, word
counts. There is deliberately no "LLM as a judge" (asking a language model
whether the answer was good), because when a judge like that says "fail" you
cannot tell whether the system answered badly or the judge marked badly. This
directory exists to tell those two apart, so the marking has to be something you
can read and reason about line by line.

Every check returns `(passed, reason)`, and the reason is what gets printed in
the results table — so a failure explains itself without anyone re-running it.
"""

import re
import unicodedata

from .cases import Case


class Grader:
    """Marks one answer against one case. Knows every name in the corpus.

    Why it needs the full list of names and not just the expected ones: to spot
    people the answer added. If the question is "who knows Python?" and the
    answer names somebody who does not, we can only call that out if we can
    recognise their name in the first place.
    """

    # The system is instructed to say plainly when the CVs do not contain an
    # answer. It phrases that differently every time, so we look for any of the
    # usual wordings rather than one exact sentence.
    _DECLINE_MARKERS: tuple[str, ...] = (
        "no candidate",
        "none of the",
        "not mentioned",
        "does not contain",
        "do not contain",
        "no information",
        "cannot answer",
        "can't answer",
        "not in the",
        "no cv",
        "nobody",
    )

    # For opinion questions there is no correct set of names, so the only thing
    # we can insist on is that the system actually answered. Fewer words than
    # this is a fragment or a dodge, not an opinion.
    _QUALITATIVE_MINIMUM_WORDS: int = 10

    def __init__(self, all_candidate_names: set[str]) -> None:
        self._all_names = all_candidate_names

    def grade(self, case: Case, answer: str, sources: list[str]) -> tuple[bool, str]:
        """Run whichever checks this case asked for, stopping at the first failure.

        `sources` is the list of CVs the backend says it used to build the
        answer, which lets us check retrieval separately from the wording.
        """
        if not answer.strip():
            return False, "empty answer"

        if case.must_decline:
            return self._grade_decline(answer)

        if case.expected_names is not None:
            # The strict check: exactly the expected people. "missing" is
            # somebody the answer forgot, "extra" is somebody it added who does
            # not belong — both are wrong, in different directions.
            mentioned_names = self.names_mentioned_in(answer)
            missing_names = case.expected_names - mentioned_names
            extra_names = mentioned_names - case.expected_names
            if missing_names or extra_names:
                return False, f"missing={sorted(missing_names)} extra={sorted(extra_names)}"

        if case.must_name:
            # The looser check: these names must appear, others are allowed. A
            # ranking answer names the winner and usually the runner-up it beat,
            # and the runner-up should not count against it.
            missing_names = case.must_name - self.names_mentioned_in(answer)
            if missing_names:
                return False, f"did not name {sorted(missing_names)}"

        if case.minimum_names:
            mentioned_names = self.names_mentioned_in(answer)
            if len(mentioned_names) < case.minimum_names:
                return (
                    False,
                    f"named {len(mentioned_names)} candidates, "
                    f"expected at least {case.minimum_names}",
                )

        if case.minimum_sources and len(set(sources)) < case.minimum_sources:
            # This checks retrieval rather than writing. If the backend only
            # fetched ten CVs for a question about all thirty, the answer is
            # already wrong no matter how well it is phrased — and this says so
            # in a way that is much easier to debug than reading the prose.
            return (
                False,
                f"retrieved {len(set(sources))} CVs, expected at least {case.minimum_sources}",
            )

        if case.must_match and not re.search(case.must_match, answer, re.IGNORECASE):
            return False, f"answer does not match {case.must_match!r}"

        for distinguishing_token in case.must_distinguish:
            if self._normalise(distinguishing_token) not in self._normalise(answer):
                return False, f"did not mention {distinguishing_token!r}"

        if case.kind == "qualitative":
            return self._grade_qualitative(answer)

        return True, "ok"

    def grade_route(self, case: Case, route: str | None) -> tuple[bool, str]:
        """Check the backend chose the strategy this question needs.

        Graded separately from the answer because the two fail for different
        reasons: a right answer down the wrong route is usually luck, and it
        tends to become a wrong answer as soon as the corpus grows.
        """
        if case.expected_route is None:
            return True, ""
        if route == case.expected_route:
            return True, ""
        return False, f"routed {route}, expected {case.expected_route}"

    def names_mentioned_in(self, answer: str) -> set[str]:
        """Which candidates the answer mentions, by full name or by surname alone.

        Names are matched as whole words. It is tempting to write
        `if name in answer`, but that matches inside other words, and it caused
        real trouble here: the corpus contains a "Ming Li", and any answer with
        the words "skills", "leading" or "Lisbon" in it contains "li". The
        grader reported five failures the system had never committed.

        The (?<!\\w) and (?!\\w) around the name are the fix — they require that
        no letter, digit or underscore sits immediately either side of the match.
        """
        normalised_answer = self._normalise(answer)
        mentioned_names = set()
        for name in self._all_names:
            name_parts = self._normalise(name).split()
            surname = [name_parts[-1]] if name_parts else []
            # Answers often switch to surname-only after the first mention
            # ("Silva has the longest tenure"), so both forms count.
            written_forms = [self._normalise(name), *surname]
            if any(
                re.search(rf"(?<!\w){re.escape(written_form)}(?!\w)", normalised_answer)
                for written_form in written_forms
            ):
                mentioned_names.add(name)
        return mentioned_names

    def _grade_decline(self, answer: str) -> tuple[bool, str]:
        """For questions the CVs cannot answer: it must say so, and name nobody.

        Both halves matter. An answer that says "no candidate lists COBOL, though
        Ana Silva works with legacy systems" has technically declined, but it has
        also put a name in front of a recruiter for a skill she does not have.
        """
        declined = any(marker in self._normalise(answer) for marker in self._DECLINE_MARKERS)
        invented_names = self.names_mentioned_in(answer)
        if not declined:
            return False, "did not decline"
        if invented_names:
            return False, f"declined but still named {sorted(invented_names)}"
        return True, "declined"

    def _grade_qualitative(self, answer: str) -> tuple[bool, str]:
        """For opinion questions: it has to answer, and to back the answer with a CV.

        Citations in this system are filenames, so looking for ".pdf" is how we
        check the opinion was drawn from the corpus rather than invented.
        """
        if len(answer.split()) < self._QUALITATIVE_MINIMUM_WORDS:
            return False, "too short to be an answer"
        if not re.search(r"\.pdf", answer):
            return False, "cited no source"
        return True, "ok"

    @staticmethod
    def _normalise(text: str) -> str:
        """Flatten text so small differences in writing do not count as differences.

        Accents are stripped (the model may write "Martinez" where the CV says
        "Martínez"), runs of whitespace become single spaces, and everything is
        lowercased. Used on both sides of every comparison in this file.
        """
        without_accents = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
        return re.sub(r"\s+", " ", without_accents).lower()
