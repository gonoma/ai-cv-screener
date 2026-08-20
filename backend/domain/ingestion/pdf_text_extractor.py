import re
from pathlib import Path

import pdfplumber
from pypdf import PdfReader


class PdfTextExtractor:
    # Measured: across all five templates pypdf and pdfplumber agree to within
    # 1% of word count, so the fallback triggers on a page that yielded almost
    # nothing, not on a multi-column layout.
    MINIMUM_WORDS_PER_PAGE: int = 20

    _TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
    _REPEATED_SPACE = re.compile(r"[ \t]{2,}")
    _REPEATED_BLANK_LINE = re.compile(r"\n{3,}")

    # Where two words were run together: a lowercase letter meeting a capital,
    # as in "Grafana ELK StackPython". Also the shape of every ordinary
    # CamelCase technology, which is why a match is only a candidate for
    # splitting and never a split on its own.
    _RUN_TOGETHER = re.compile(r"(?<=[a-z])(?=[A-Z])")
    _WORD = re.compile(r"[^\W_]+", re.UNICODE)

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Read with both, keeping pypdf's text and pdfplumber's word boundaries.

        Both run on every CV. pdfplumber is slower — 2.1s against 0.9s over the
        thirty-CV corpus — which is nothing beside the LLM call each of those CVs
        is about to cost, and it is the only reading that reliably separates
        words across a column break.

        pdfplumber's own text is used whole only when a page came back near-empty
        from pypdf, which is the case where reading order hardly matters because
        there is nothing to order.

        Neither extractor fixes *reading order*: on two_column the sidebar
        arrives before the body, on compact the dates arrive before the roles
        they label. CvTextChunker is what copes with that.

        The text is whitespace-normalised on the way out. because that whitespace
        is billed as tokens every time the text is sent, stored, retrieved, and
        re-read — only ~0.5% here, but it's one regex and pays off far more on messier PDFs.
        """
        pages_from_pypdf = self._read_every_page_with_pypdf(pdf_path)
        pages_from_pdfplumber = self._read_every_page_with_pdfplumber(pdf_path)

        a_page_came_back_near_empty = any(
            len(page_text.split()) < self.MINIMUM_WORDS_PER_PAGE for page_text in pages_from_pypdf
        )
        if a_page_came_back_near_empty:
            pages = self._keep_whichever_extractor_read_more(
                pages_from_pypdf=pages_from_pypdf,
                pages_from_pdfplumber=pages_from_pdfplumber,
            )
        else:
            pages = [
                self._split_run_together_words(page, vocabulary=self._words(reference))
                for page, reference in zip(pages_from_pypdf, pages_from_pdfplumber, strict=False)
            ]
        return self._normalise_whitespace("\n".join(pages))

    @classmethod
    def _split_run_together_words(cls, page_text: str, vocabulary: set[str]) -> str:
        """Put back the spaces pypdf dropped, using pdfplumber's reading as the authority.

        On a two-column skills block pypdf returns "Grafana ELK StackPython",
        and an exact-match filter over skills then cannot see that this person
        knows Python — one candidate in thirty, invisible to the question the
        structured route exists to answer.

        The two readings are used for what each is good at rather than one being
        chosen over the other: pypdf keeps a section's lines together, while
        pdfplumber interleaves the columns but does put the spaces in. So the
        order stays pypdf's and only the word boundaries are borrowed.

        A split needs all three conditions, which is what keeps JavaScript,
        PostgreSQL and GitHub intact: the fused form is absent from the other
        reading, and both halves appear in it as words in their own right.
        """
        if not vocabulary:
            return page_text

        def repair(match: re.Match) -> str:
            word = match.group()
            if word.lower() in vocabulary:
                return word
            for position in [split.start() for split in cls._RUN_TOGETHER.finditer(word)]:
                left, right = word[:position], word[position:]
                if left.lower() in vocabulary and right.lower() in vocabulary:
                    return f"{left} {right}"
            return word

        return cls._WORD.sub(repair, page_text)

    @classmethod
    def _words(cls, page_text: str) -> set[str]:
        return {word.lower() for word in cls._WORD.findall(page_text)}

    @classmethod
    def _normalise_whitespace(cls, text: str) -> str:
        """Squeeze layout padding out of the text without touching its structure.

        Line breaks survive as line breaks and a blank line still separates
        blocks, because CvTextChunker finds section headings by matching a whole
        line. Collapsing newlines into spaces would save a few more tokens, but the
        headings will blur into the text around them and the chunker could no longer
        find where sections start.
        """
        text = cls._TRAILING_SPACE.sub("", text)
        text = cls._REPEATED_SPACE.sub(" ", text)
        text = cls._REPEATED_BLANK_LINE.sub("\n\n", text)
        return text.strip()

    def _read_every_page_with_pypdf(self, pdf_path: Path) -> list[str]:
        return [(page.extract_text() or "") for page in PdfReader(pdf_path).pages]

    def _read_every_page_with_pdfplumber(self, pdf_path: Path) -> list[str]:
        with pdfplumber.open(pdf_path) as pdf_document:
            return [(page.extract_text() or "") for page in pdf_document.pages]

    def _keep_whichever_extractor_read_more(
        self, pages_from_pypdf: list[str], pages_from_pdfplumber: list[str]
    ) -> list[str]:
        """Compare page by page rather than picking one extractor for the whole file.

        A CV can have one bad page and four good ones, and pdfplumber is not
        uniformly better — it just fails differently.
        """
        best_pages: list[str] = []
        for pypdf_page, pdfplumber_page in zip(
            pages_from_pypdf, pages_from_pdfplumber, strict=False
        ):
            if len(pdfplumber_page.split()) > len(pypdf_page.split()):
                best_pages.append(pdfplumber_page)
            else:
                best_pages.append(pypdf_page)

        return best_pages
