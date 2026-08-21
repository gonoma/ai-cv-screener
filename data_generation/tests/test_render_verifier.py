"""The renderer's own guard: a template that loses content has to fail the render.

Two real regressions motivate these, both found in the committed corpus rather
than imagined: `compact.html` floated its date ranges, which collected them into
a single line beside the last job, and both `compact.html` and
`sidebar_right.html` clipped whatever ran past the bottom of the page instead of
paginating it, so the longest CVs lost their schools and languages outright.
"""

import pathlib

import pytest

from data_generation.models.cv_records import Candidate, Qualification, Role
from data_generation.rendering import CvPdfRenderer
from data_generation.rendering.render_verifier import RenderedCvIncomplete, RenderVerifier

TEMPLATES = CvPdfRenderer.TEMPLATE_FILENAMES


def build_candidate(role_count: int) -> Candidate:
    """A candidate whose length is the variable, because length is what broke.

    Four bullets a role is the upper end of what the content builder writes, so
    a long career here fills the page the same way the real records did.
    """
    return Candidate(
        name="Renata Oliveira",
        headline="Principal Platform Engineer",
        email="renata.oliveira@example.com",
        phone="+351 912 345 678",
        location="Porto, Portugal",
        presentation="a woman",
        appearance="in her forties, olive skin, dark curly hair",
        summary="Platform engineer with a long record of running production systems.",
        skills=[f"Skill {index}" for index in range(12)],
        languages=["Portuguese (native)", "English (C2)", "Spanish (B2)"],
        experience=[
            Role(
                company=f"Company {index}",
                role=f"Engineer Level {index}",
                location="Porto, Portugal",
                start_year=2000 + index * 2,
                end_year=2002 + index * 2 if index < role_count - 1 else None,
                bullets=[f"Delivered project {index}.{bullet} on time." for bullet in range(4)],
            )
            for index in range(role_count)
        ],
        education=[
            Qualification(
                institution="Universidade do Porto",
                short_name="UP",
                degree="MSc Computer Science",
                field="Computer Science",
                graduation_year=2000,
            )
        ],
    )


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize("role_count", [2, 6])
def test_every_template_keeps_the_whole_record(
    template: str, role_count: int, tmp_path: pathlib.Path
) -> None:
    """Short and long careers alike, on every template.

    The long case is the one that matters: six roles overflow one A4 page, which
    is exactly the condition under which a float or a clipping box used to drop
    the sections that came last.
    """
    output_path = tmp_path / "cv.pdf"
    CvPdfRenderer().render_to_pdf(
        candidate=build_candidate(role_count),
        photo_data_uri=None,
        template=template,
        output_path=output_path,
    )

    assert output_path.exists()


def test_a_clipped_layout_is_rejected(tmp_path: pathlib.Path) -> None:
    """The original `compact.html` bug, reintroduced on purpose.

    Guards the guard: if `verify` ever stops reading the PDF back, every test
    above would keep passing while the corpus quietly rotted again.

    Five roles, and not four or six, because the bug needs the two-up block to
    straddle the page boundary. Shorter and it fits on page one; longer and it
    is pushed wholly onto page two, where it has a full page and survives. That
    narrow window is why the fault reached the committed corpus at all: it hit
    one of the six compact CVs and left the other five looking perfect.
    """
    candidate = build_candidate(role_count=5)
    template_path = (
        pathlib.Path(CvPdfRenderer._TEMPLATE_DIRECTORY) / "compact.html"  # noqa: SLF001
    )
    broken = template_path.read_text(encoding="utf-8").replace(
        ".two-up { display: flex; gap: 4%; }",
        ".two-up { overflow: hidden; }\n  .two-up .col { float: left; width: 48%; }",
    )
    broken_path = tmp_path / "broken.html"
    broken_path.write_text(broken, encoding="utf-8")

    renderer = CvPdfRenderer()
    renderer._environment.loader.searchpath.append(str(tmp_path))  # noqa: SLF001

    with pytest.raises(RenderedCvIncomplete, match="missing from the page"):
        renderer.render_to_pdf(
            candidate=candidate,
            photo_data_uri=None,
            template="broken.html",
            output_path=tmp_path / "cv.pdf",
        )


def test_detached_years_are_rejected() -> None:
    """Presence is not enough: a range far from its role is still a broken page.

    The float bug put every date range on the page, just in the wrong place, so
    a check that only asked "is 2016 - 2018 here?" passed it.
    """
    candidate = build_candidate(role_count=2)
    role = candidate.experience[0]
    detached = (
        f"{candidate.name} {role.role} {role.company} "
        + "filler " * 60
        + f"{role.start_year} - {role.end_year}"
    )

    problems = RenderVerifier()._detached_years(  # noqa: SLF001
        candidate,
        RenderVerifier._squash(detached),  # noqa: SLF001
    )

    assert any("detached from their role" in problem for problem in problems)
