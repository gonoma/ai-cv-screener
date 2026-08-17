import re
import unicodedata
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

from ..models.cv_records import Candidate, Role


class CvPdfRenderer:
    _TEMPLATE_DIRECTORY: Path = Path(__file__).parent / "templates"

    TEMPLATE_FILENAMES: tuple[str, ...] = (
        "single_column.html",
        "two_column.html",
        "compact.html",
        "timeline.html",
        "sidebar_right.html",
    )

    def __init__(self) -> None:
        self._environment = Environment(
            loader=FileSystemLoader(self._TEMPLATE_DIRECTORY),
            autoescape=select_autoescape(["html"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._environment.filters["date_range"] = self._format_date_range

    def template_for_index(self, index: int) -> str:
        """Cycle the templates by position rather than choosing one at random."""
        return self.TEMPLATE_FILENAMES[index % len(self.TEMPLATE_FILENAMES)]

    @staticmethod
    def pdf_filename_for(name: str) -> str:
        ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
        filename_slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
        return f"{filename_slug}.pdf"

    def render_to_pdf(
        self, candidate: Candidate, photo_data_uri: str | None, template: str, output_path: Path
    ) -> None:
        rendered_html = self._environment.get_template(template).render(
            candidate=candidate, photo=photo_data_uri
        )
        HTML(string=rendered_html).write_pdf(output_path)

    @staticmethod
    def _format_date_range(role: Role) -> str:
        return f"{role.start_year} - {role.end_year or 'Present'}"
