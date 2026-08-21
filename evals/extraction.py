"""Marks how well ingestion read the CVs, by comparing it to the answer key.

    make eval-extraction

Two evals live in this directory, and they measure different halves of the
system. `run.py` measures the answers: it asks the backend real questions, which
means a running server and one language-model call per question. This file
measures the *reading* step that happens before any question is asked — did we
correctly pull each candidate's role, years of experience, skills and
universities out of their PDF?

It is free and needs nothing running, because ingestion already wrote what the
model returned for each CV into `data/extractions/*.json`. We re-apply the same
derivations the ingestion pipeline applies (`candidate_facts`) and compare the
result to `data/ground_truth.json`. That is the same data the database holds, so
this measures the database without needing one.

Why bother measuring this separately: extraction mistakes are invisible further
down. When the model was asked outright for "years of experience" it returned
numbers that were plausible, wrong, and completely undetectable — nothing
downstream can argue with a well-formed integer. This eval is what caught that,
and what now stops it coming back.
"""

import json
import re
import sys
from pathlib import Path

from backend.domain import candidate_facts

from . import REPO_ROOT
from .cases import AnswerKey


class ExtractionScorer:
    """Compares every parsed CV against the answer key, field by field."""

    _EXTRACTIONS: Path = REPO_ROOT / "data" / "extractions"

    # The pass marks. Below these, the data in the database is not good enough to
    # answer questions from. They were set by measuring the corpus as committed
    # (100% / 87% / 99% / 100%) and then sitting a little under it, so that
    # regenerating the corpus with different people still passes, while a real
    # regression does not.
    #
    # They differ because the fields differ:
    #   - years is arithmetic over dates, not reading, so anything but 100% means
    #     our own calculation is wrong — hence the 1.0.
    #   - a job title is prose. A CV printing "Solutions Architect Intern" in the
    #     body under a "Solutions Architect" headline is a disagreement about
    #     wording, not a failure to read the page.
    #   - skill recall sits below what we measure because the two skills we miss
    #     are not the same two on every run: the misses move between CVs when a
    #     record is re-extracted, so the rate is stable and the identity is not.
    #   - institution recall used to sit at 0.85 to absorb PDFs that had dropped
    #     an Education section during rendering. They cannot any more — the
    #     renderer reads every CV back and refuses to write one that lost content
    #     (see RenderVerifier) — so the only misses left would be the extractor
    #     misreading a name that is demonstrably on the page. 0.95 leaves room for
    #     three of those and still catches a clipped section, which loses a whole
    #     school list at once.
    _THRESHOLDS: dict[str, float] = {
        "years exact": 1.0,
        "role exact": 0.85,
        "skill recall": 0.95,
        "institution recall": 0.95,
    }

    # Enough misses to see a pattern, not so many that the table scrolls off.
    _MISSES_PRINTED: int = 15
    _RULE_WIDTH: int = 60

    _EXIT_ALL_PASSED: int = 0
    _EXIT_BELOW_THRESHOLD: int = 1
    _EXIT_COULD_NOT_RUN: int = 2

    def __init__(self) -> None:
        self._answer_key = AnswerKey()
        # For each metric: [how many we got right, how many there were]. The two
        # exact-match metrics count one per CV; the two recall metrics count one
        # per list entry (a CV with twelve skills contributes twelve). Both are
        # read as a single percentage at the end.
        self._scores: dict[str, list[int]] = {name: [0, 0] for name in self._THRESHOLDS}
        # One human-readable line per thing we got wrong, printed under the table.
        self._misses: list[str] = []

    def run(self) -> int:
        extraction_files = sorted(self._EXTRACTIONS.glob("*.json"))
        if not extraction_files:
            print(f"no extractions in {self._EXTRACTIONS} — run `make ingest` first")
            return self._EXIT_COULD_NOT_RUN

        expected_by_file = self._answer_key.candidates_by_source_file()
        for path in extraction_files:
            cached_extraction = json.loads(path.read_text(encoding="utf-8"))
            source_file = cached_extraction["source_file"]
            expected = expected_by_file.get(source_file)
            if expected is None:
                # An extraction for a CV the key has never heard of. Usually it
                # means the corpus was regenerated but old extractions were left
                # behind, so it is worth saying rather than skipping quietly.
                self._misses.append(f"{source_file}: not in the answer key")
                continue
            self._score_cv(
                source_file=source_file,
                extracted=self._derived_database_row(cached_extraction["candidate_info"]),
                expected=expected,
            )

        return self._report(len(extraction_files))

    def _score_cv(self, source_file: str, extracted: dict, expected: dict) -> None:
        """Compare one CV's four fields and record what matched and what did not."""
        for label, extracted_value, expected_value in (
            ("years exact", extracted["years"], expected["years_experience"]),
            ("role exact", extracted["role"], expected["current_role"]),
        ):
            # Years is a number and compares as one. A title is prose, so it is
            # compared with spacing and capitalisation flattened away first.
            matched = (
                extracted_value == expected_value
                if label == "years exact"
                else self._normalise(extracted_value) == self._normalise(expected_value)
            )
            self._scores[label][0] += bool(matched)
            self._scores[label][1] += 1
            if not matched:
                self._misses.append(
                    f"{source_file}: {label} — got {extracted_value!r}, key {expected_value!r}"
                )

        # Skills and universities are lists, so the question is not "is it
        # right?" but "how much of it did we find?" — that is what recall means.
        for label, field_name in (
            ("skill recall", "skills"),
            ("institution recall", "institutions"),
        ):
            recalled, lost = self._count_recalled(expected[field_name], extracted[field_name])
            self._scores[label][0] += recalled
            self._scores[label][1] += len(expected[field_name])
            if lost:
                self._misses.append(f"{source_file}: {label} — lost {lost}")

    def _derived_database_row(self, extraction: dict) -> dict:
        """Rebuild the database row that ingestion would write from this extraction.

        The model does not hand us a finished row. It returns the raw pieces (a
        list of positions, a job title that may repeat the company name), and the
        ingestion pipeline derives the stored values from them — the current job,
        the title with the employer stripped off, the total years worked. We call
        the very same `candidate_facts` helpers here, so what gets marked is what
        the database really holds, not an idealised version of it.
        """
        positions = extraction.get("positions") or []
        current_position = candidate_facts.current_position(positions) or {}
        company = extraction.get("current_company") or current_position.get("company")
        return {
            "name": extraction.get("name"),
            "role": candidate_facts.role_without_company(
                role=extraction.get("current_role") or current_position.get("role"),
                company=company,
            ),
            "years": candidate_facts.career_years(positions, as_of=self._answer_key.corpus_year),
            "skills": extraction.get("skills") or [],
            "institutions": extraction.get("institutions") or [],
        }

    def _report(self, extractions_scored: int) -> int:
        """Print the table, then every miss, and return the exit code."""
        print(f"{extractions_scored} extractions scored against the answer key\n")
        print(f"{'metric':20} {'score':12} {'floor':7} result")
        print("-" * self._RULE_WIDTH)

        below_threshold = False
        for label, (hits, total) in self._scores.items():
            rate = hits / total if total else 0.0
            passed = rate >= self._THRESHOLDS[label]
            below_threshold |= not passed
            print(
                f"{label:20} {f'{hits}/{total}':12} {self._THRESHOLDS[label]:<7.2f} "
                f"{'PASS' if passed else 'FAIL'} ({rate:.0%})"
            )

        if self._misses:
            print(f"\n{len(self._misses)} field(s) off:")
            for miss in self._misses[: self._MISSES_PRINTED]:
                print(f"  {miss}")
            if len(self._misses) > self._MISSES_PRINTED:
                print(f"  ... and {len(self._misses) - self._MISSES_PRINTED} more")

        return self._EXIT_BELOW_THRESHOLD if below_threshold else self._EXIT_ALL_PASSED

    @classmethod
    def _count_recalled(cls, expected: list[str], extracted: list[str]) -> tuple[int, list[str]]:
        """How many of the key's entries we found, and which ones we lost.

        Matching is not a plain string comparison, because the two sides format
        the same fact differently. The key writes a university as one string,
        "Universidad Politécnica de Madrid (UPM)", while extraction is asked to
        return the full name and the abbreviation as separate entries — that is
        deliberate, and it is what makes a question about "UPM" findable later.

        So an entry counts as found when each of its parts appears somewhere in
        what we extracted. Comparing whole strings instead marked plenty of
        correct extractions as losses, which buried the two CVs that had genuinely
        come back empty.
        """
        normalised_extraction = {cls._normalise(entry) for entry in extracted}
        lost = [entry for entry in expected if not cls._is_recalled(entry, normalised_extraction)]
        return len(expected) - len(lost), lost

    @classmethod
    def _is_recalled(cls, expected: str, extracted: set[str]) -> bool:
        """True when every part of this entry turned up in the extraction."""
        return all(cls._part_appears(part, extracted) for part in cls._entry_parts(expected))

    @classmethod
    def _entry_parts(cls, expected: str) -> list[str]:
        """Split "Full Name (ABBR)" into the separate things a CV actually prints.

        "Universidad Politécnica de Madrid (UPM) – MSc" becomes
        ["universidad politécnica de madrid", "upm"]: the text in brackets is
        pulled out as its own part, and anything after a dash (a degree, a year)
        is dropped, since extraction is not asked to keep it.
        """
        normalised = cls._normalise(expected)
        abbreviations = re.findall(r"\(([^)]*)\)", normalised)
        full_name = re.sub(r"\([^)]*\)", " ", normalised)
        full_name = re.split(r"\s+[–—-]\s+", full_name)[0]
        return [
            stripped_part for part in [full_name, *abbreviations] if (stripped_part := part.strip())
        ]

    @staticmethod
    def _part_appears(part: str, extracted: set[str]) -> bool:
        """Does this part appear in the extraction, in either direction?

        Either side may be the longer one — the key's "TU Delft" against an
        extracted "Delft University of Technology (TU Delft)", or the reverse —
        so we accept a match whichever string contains the other.
        """
        return any(part in entry or entry in part for entry in extracted)

    @staticmethod
    def _normalise(text: str) -> str:
        """Flatten spacing and capitalisation so only real differences count."""
        return " ".join(str(text).split()).strip().lower()


def main() -> int:
    return ExtractionScorer().run()


if __name__ == "__main__":
    sys.exit(main())
