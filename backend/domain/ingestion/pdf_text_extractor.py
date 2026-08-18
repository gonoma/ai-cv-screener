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

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Read with pypdf, and only reach for pdfplumber if a page came back near-empty.

        pdfplumber is several times slower, so running both on every CV would
        cost minutes across a thirty-document corpus for no measured gain.

        Neither extractor fixes *reading order*: on two_column the sidebar
        arrives before the body, on compact the dates arrive before the roles
        they label. CvTextChunker is what copes with that.

        The text is whitespace-normalised on the way out. because that whitespace
        is billed as tokens every time the text is sent, stored, retrieved, and
        re-read — only ~0.5% here, but it's one regex and pays off far more on messier PDFs.
        """
        pages_from_pypdf = self._read_every_page_with_pypdf(pdf_path)
        a_page_came_back_near_empty = any(
            len(page_text.split()) < self.MINIMUM_WORDS_PER_PAGE for page_text in pages_from_pypdf
        )
        if not a_page_came_back_near_empty:
            return self._normalise_whitespace("\n".join(pages_from_pypdf))

        pages_from_pdfplumber = self._read_every_page_with_pdfplumber(pdf_path)
        return self._normalise_whitespace(
            "\n".join(
                self._keep_whichever_extractor_read_more(
                    pages_from_pypdf=pages_from_pypdf,
                    pages_from_pdfplumber=pages_from_pdfplumber,
                )
            )
        )

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
