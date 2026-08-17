import re

from ...data.models import CvChunk


class CvTextChunker:
    CHUNK_CHARS: int = 900
    CHUNK_OVERLAP: int = 150

    SECTION_HEADINGS: list[str] = [
        "profile",
        "summary",
        "about",
        "experience",
        "work experience",
        "employment",
        "education",
        "skills",
        "technical skills",
        "languages",
        "contact",
    ]

    def __init__(self) -> None:
        headings_longest_first = sorted(self.SECTION_HEADINGS, key=len, reverse=True)
        alternatives = "|".join(re.escape(heading) for heading in headings_longest_first)
        self._heading_pattern = re.compile(
            r"^\s*(" + alternatives + r")\s*$", re.IGNORECASE | re.MULTILINE
        )

    def chunk_cv_text(self, cv_text: str) -> list[CvChunk]:
        chunks: list[CvChunk] = []
        for heading, section_text in self.split_into_sections(cv_text):
            for window in self.split_into_overlapping_windows(section_text):
                if window.strip():
                    chunks.append(CvChunk(section=heading, content=window.strip()))
        return chunks

    def split_into_sections(self, cv_text: str) -> list[tuple[str | None, str]]:
        """Cut the text at recognised headings, returning (heading, body) pairs.

        Two cases are easy to miss. The text before the first heading has no
        heading of its own but is not throwaway — it is where the candidate's
        name and contact block usually land — so it is emitted with a heading of
        None rather than dropped. And a CV whose headings are unrecognised is
        returned whole, so it still chunks, just without structure.
        """
        heading_matches = list(self._heading_pattern.finditer(cv_text))
        if not heading_matches:
            return [(None, cv_text)]

        sections: list[tuple[str | None, str]] = []
        text_before_first_heading = cv_text[: heading_matches[0].start()].strip()
        if text_before_first_heading:
            sections.append((None, text_before_first_heading))

        for heading_match, next_heading_match in zip(
            heading_matches, heading_matches[1:] + [None], strict=False
        ):
            section_ends_at = next_heading_match.start() if next_heading_match else len(cv_text)
            section_text = cv_text[heading_match.end() : section_ends_at].strip()
            if section_text:
                sections.append((heading_match.group(1).strip().title(), section_text))
        return sections

    def split_into_overlapping_windows(self, section_text: str) -> list[str]:
        """Cut a long section into windows that deliberately repeat their edges.

        Each window advances by less than its own width, so the last
        CHUNK_OVERLAP characters of one window are the first of the next. A fact
        that straddles a boundary therefore survives whole in one of the two,
        instead of being halved into both and retrievable from neither.
        """
        if len(section_text) <= self.CHUNK_CHARS:
            return [section_text]

        characters_to_advance = self.CHUNK_CHARS - self.CHUNK_OVERLAP
        windows = (
            section_text[window_start : window_start + self.CHUNK_CHARS]
            for window_start in range(0, len(section_text), characters_to_advance)
        )
        return [window for window in windows if window.strip()]
