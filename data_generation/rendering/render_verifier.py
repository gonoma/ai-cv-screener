import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from ..models.cv_records import Candidate


class RenderedCvIncomplete(AssertionError):
    """A rendered PDF is missing content the record it was built from put into it.

    Raised at generation time, on purpose. A CV whose Education section fell off
    the page still looks like a CV, and every downstream stage accepts it: the
    extractor reads what is there and reports it honestly, the answer key still
    lists the school nobody can see, and the eval books the gap as an extraction
    failure. The renderer is the only stage that knows both what went in and what
    came out, so it is the only place the loss is cheap to notice.
    """


class RenderVerifier:
    """Reads a freshly written PDF back and checks the record survived the layout.

    Two failures motivate this, both silent, both introduced by CSS rather than
    by any model:

    * content clipped away entirely, when a float or an `overflow: hidden` box
      ran past the bottom of the page and WeasyPrint dropped the remainder
      instead of paginating it;
    * content present but detached from what it describes, when a float pulled
      every date range out of its role and left them in a heap elsewhere on the
      page. Presence alone will not catch that, so the year check is a proximity
      check: a range has to land near the job it belongs to.
    """

    # Both extractors the backend uses read the content stream, so this reads the
    # PDF the same way the pipeline will. Distances below are measured in the
    # whitespace-stripped text, where a role line runs 40-90 characters; 200 is
    # comfortably wider than any template's role block and far narrower than the
    # page-length gap a detached float leaves behind.
    _MAX_YEARS_DISTANCE: int = 200

    _DASHES: dict[int, str] = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")

    def verify(self, candidate: Candidate, pdf_path: Path) -> None:
        text = self._squash(self._read_text(pdf_path))

        problems = self._missing_content(candidate, text) + self._detached_years(candidate, text)
        if problems:
            listed = "\n  - ".join(problems)
            raise RenderedCvIncomplete(
                f"{pdf_path.name}: the layout lost content that was rendered into it."
                f"\n  - {listed}"
                "\n  The template, not the record, is at fault: check for a float or an"
                "\n  `overflow: hidden` box running past the bottom of the page."
            )

    def _missing_content(self, candidate: Candidate, text: str) -> list[str]:
        """Every string the record guarantees a template puts on the page.

        Skills are left out: `compact.html` flows them through CSS columns and
        `two_column.html` through chips, both of which are allowed to reorder
        them. Nothing here depends on order, only on being present at all.
        """
        required: list[tuple[str, str]] = [("name", candidate.name)]
        required += [("institution", name) for name in candidate.institution_names()]
        required += [("language", language) for language in candidate.languages]
        for role in candidate.experience:
            required += [("company", role.company), ("role", role.role)]

        return [
            f"{label} missing from the page: {value!r}"
            for label, value in required
            if self._squash(value) not in text
        ]

    def _detached_years(self, candidate: Candidate, text: str) -> list[str]:
        """Each role's date range has to sit near the role it dates.

        Anchored on the company rather than the job title because two roles at
        one employer share a title far more often than two employers share a name.
        """
        problems = []
        for role in candidate.experience:
            span = self._squash(f"{role.start_year} - {role.end_year or 'Present'}")
            anchors = self._offsets(text, self._squash(role.company))
            spans = self._offsets(text, span)
            if not spans:
                problems.append(f"date range missing from the page: {span!r} ({role.company})")
                continue
            if not anchors:
                continue  # already reported as a missing company

            distance = min(abs(s - a) for a in anchors for s in spans)
            if distance > self._MAX_YEARS_DISTANCE:
                problems.append(
                    f"date range {span!r} sits {distance} characters from {role.company!r}; "
                    "the years have been detached from their role"
                )

        return problems

    @staticmethod
    def _read_text(pdf_path: Path) -> str:
        reader = PdfReader(pdf_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    @classmethod
    def _squash(cls, text: str) -> str:
        """Compare on letters alone.

        WeasyPrint writes the thin and non-breaking spaces the records contain,
        wraps lines wherever the column ends, and hyphenates with U+2011; none of
        that is a difference in content. Dropping whitespace entirely also means
        a name split across two lines still matches.
        """
        folded = unicodedata.normalize("NFKC", text).translate(cls._DASHES).casefold()
        return re.sub(r"\s+", "", folded)

    @staticmethod
    def _offsets(haystack: str, needle: str) -> list[int]:
        if not needle:
            return []
        return [match.start() for match in re.finditer(re.escape(needle), haystack)]
