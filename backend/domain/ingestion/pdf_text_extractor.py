from pathlib import Path

import pdfplumber
from pypdf import PdfReader


class PdfTextExtractor:
    # Measured: across all five templates pypdf and pdfplumber agree to within
    # 1% of word count, so the fallback triggers on a page that yielded almost
    # nothing, not on a multi-column layout.
    MINIMUM_WORDS_PER_PAGE: int = 20

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Read with pypdf, and only reach for pdfplumber if a page came back near-empty.

        pdfplumber is several times slower, so running both on every CV would
        cost minutes across a thirty-document corpus for no measured gain.

        Neither extractor fixes *reading order*: on two_column the sidebar
        arrives before the body, on compact the dates arrive before the roles
        they label. CvTextChunker is what copes with that.
        """
        pages_from_pypdf = self._read_every_page_with_pypdf(pdf_path)
        a_page_came_back_near_empty = any(
            len(page_text.split()) < self.MINIMUM_WORDS_PER_PAGE for page_text in pages_from_pypdf
        )
        if not a_page_came_back_near_empty:
            return "\n".join(pages_from_pypdf).strip()

        pages_from_pdfplumber = self._read_every_page_with_pdfplumber(pdf_path)
        return "\n".join(
            self._keep_whichever_extractor_read_more(pages_from_pypdf, pages_from_pdfplumber)
        ).strip()

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
        return [
            pdfplumber_page_text
            if len(pdfplumber_page_text.split()) > len(pypdf_page_text.split())
            else pypdf_page_text
            for pypdf_page_text, pdfplumber_page_text in zip(
                pages_from_pypdf, pages_from_pdfplumber, strict=False
            )
        ]
