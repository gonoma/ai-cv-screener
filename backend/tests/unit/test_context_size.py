"""What goes into the prompt, which is what a token budget actually buys.

These assert on the *content* of the retrieved context rather than on a token
count: a threshold would need rewriting every time a prompt is tuned, while the
properties here — don't send fields nobody asked about, don't send the same
sentence twice — are the reasons the count is what it is.
"""

from backend.data.models import QueryRoute
from backend.domain.query import ContextRetriever

CANDIDATES = [
    ("ada.pdf", "Ada Lovelace", ["Python", "Kubernetes"]),
    ("alan.pdf", "Alan Turing", ["Python"]),
]


class FakeResult:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def fetchall(self) -> list:
        return self._rows


class RecordingConnection:
    """Returns fixed rows and remembers the SQL, since the projection is the point."""

    def __init__(self, rows: list) -> None:
        self.rows = rows
        self.statements: list[str] = []

    def execute(self, sql: str, parameters=None) -> FakeResult:
        self.statements.append(sql)
        return FakeResult(self.rows)


def _retriever(rows: list) -> tuple[ContextRetriever, RecordingConnection]:
    connection = RecordingConnection(rows)
    built = ContextRetriever.__new__(ContextRetriever)
    built.connection = connection
    return built, connection


def test_a_skills_question_does_not_send_employers_and_universities() -> None:
    rows = [(source, name, skills) for source, name, skills in CANDIDATES]
    retriever, connection = _retriever(rows)

    context = retriever._retrieve_all_candidates_matching_filters(
        QueryRoute(route="structured", skills=["Python"])
    )

    assert "institutions" not in connection.statements[0]
    assert "current_company" not in connection.statements[0]
    assert "Ada Lovelace" in context.text and "Alan Turing" in context.text
    assert context.source_files == ["ada.pdf", "alan.pdf"]


def test_a_years_question_sends_years_and_not_skills() -> None:
    retriever, connection = _retriever([("ada.pdf", "Ada Lovelace", 9)])

    retriever._retrieve_all_candidates_matching_filters(
        QueryRoute(route="structured", minimum_years_experience=5)
    )

    assert "years_experience" in connection.statements[0]
    assert "skills" not in connection.statements[0].split("FROM")[0]


def test_an_unfiltered_listing_still_sends_the_whole_record() -> None:
    """The one case where the full record is the answer, and the rarest one."""
    retriever, connection = _retriever([("ada.pdf", "Ada Lovelace", "Engineer", "Acme", 9, [], [])])

    retriever._retrieve_all_candidates_matching_filters(QueryRoute(route="structured"))

    selected = connection.statements[0].split("FROM")[0]
    assert all(column in selected for column in ("role", "current_company", "institutions"))


def test_the_deliberate_chunk_overlap_is_not_sent_twice() -> None:
    """Overlap is a retrieval property; reading a CV back in order it is pure duplication."""
    rows = [
        ("ada.pdf", "Ada Lovelace", "Experience", "led the analytical engine team in London"),
        ("ada.pdf", "Ada Lovelace", "Experience", "team in London and wrote the first program"),
    ]

    trimmed = ContextRetriever._drop_repeated_overlap(rows)

    assert trimmed[1][3] == "and wrote the first program"
    assert "".join(row[3] for row in trimmed).count("team in London") == 1


def test_overlap_is_only_dropped_within_one_document_and_section() -> None:
    """Two people can end and start on the same words without it being an overlap."""
    rows = [
        ("ada.pdf", "Ada Lovelace", "Skills", "Python"),
        ("alan.pdf", "Alan Turing", "Skills", "Python"),
    ]
    assert ContextRetriever._drop_repeated_overlap(rows) == rows


def test_chunks_that_do_not_overlap_are_left_alone() -> None:
    rows = [
        ("ada.pdf", "Ada Lovelace", "Skills", "Python and Kubernetes"),
        ("ada.pdf", "Ada Lovelace", "Skills", "Terraform and Go"),
    ]
    assert ContextRetriever._drop_repeated_overlap(rows) == rows


def test_a_chunk_wholly_contained_in_its_predecessor_is_dropped() -> None:
    rows = [
        ("ada.pdf", "Ada Lovelace", "Skills", "Python and Kubernetes"),
        ("ada.pdf", "Ada Lovelace", "Skills", "Kubernetes"),
    ]
    assert len(ContextRetriever._drop_repeated_overlap(rows)) == 1
