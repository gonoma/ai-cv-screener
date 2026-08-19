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


def _chunks(*source_files: str) -> list:
    return [
        (source, source[:-4].title(), "Experience", f"bullet {i}")
        for i, source in enumerate(source_files)
    ]


def test_one_cv_cannot_occupy_the_whole_context() -> None:
    """Whoever writes the most quantified bullets would otherwise win on volume alone."""
    retriever, _ = _retriever([])
    retriever._TOP_K = 4

    kept = retriever._spread_across_candidates(
        _chunks("ada.pdf", "ada.pdf", "ada.pdf", "ada.pdf", "alan.pdf", "grace.pdf")
    )

    assert [row[0] for row in kept] == ["ada.pdf", "ada.pdf", "alan.pdf", "grace.pdf"]


def test_the_cap_keeps_the_ranking_it_was_given() -> None:
    """Chunks are dropped, never reordered: nearest-first is still the ranking sent."""
    retriever, _ = _retriever([])
    retriever._TOP_K = 3

    kept = retriever._spread_across_candidates(_chunks("ada.pdf", "alan.pdf", "ada.pdf"))

    assert [row[0] for row in kept] == ["ada.pdf", "alan.pdf", "ada.pdf"]


def test_a_short_ranking_is_returned_whole() -> None:
    """A corpus smaller than the cap must not come back empty-handed."""
    retriever, _ = _retriever([])
    assert len(retriever._spread_across_candidates(_chunks("ada.pdf", "alan.pdf"))) == 2


def test_the_roster_states_the_seniority_a_chunk_does_not_carry() -> None:
    retriever, _ = _retriever([("Ada Lovelace", "ada.pdf", "Staff Engineer", 9)])

    roster = retriever._roster(_chunks("ada.pdf"))

    assert "Ada Lovelace (ada.pdf): Staff Engineer, 9y experience" in roster


def test_a_field_the_extraction_missed_is_left_out_of_the_roster() -> None:
    """An empty role must not render as a stray comma the model has to interpret."""
    retriever, _ = _retriever([("Ada Lovelace", "ada.pdf", "", 9)])

    assert "Ada Lovelace (ada.pdf): 9y experience" in retriever._roster(_chunks("ada.pdf"))


# --- superlatives: ordered in SQL rather than argued out of prose -----------

RANKED = [
    # source_file, name, ranked number, role, company, positions, rows matched
    (
        "ada.pdf",
        "Ada Lovelace",
        11,
        "Staff Engineer",
        "Steady Ltd",
        [
            {"role": "Analyst", "company": "Early Co", "start_year": 2012, "end_year": 2015},
            {
                "role": "Staff Engineer",
                "company": "Steady Ltd",
                "start_year": 2015,
                "end_year": None,
            },
        ],
        2,
    ),
    ("alan.pdf", "Alan Turing", 4, "Contractor", "Short Ltd", [], 2),
]


def test_a_tenure_question_orders_on_the_tenure_column() -> None:
    retriever, connection = _retriever(RANKED)

    context = retriever._retrieve_ranked_candidates(
        QueryRoute(route="structured", ranking="tenure")
    )

    assert "ORDER BY longest_tenure_years DESC NULLS LAST" in connection.statements[0]
    assert context.source_files == ["ada.pdf", "alan.pdf"]


def test_a_tenure_answer_names_the_job_that_produced_the_number() -> None:
    """A number with no job attached is a ranking the reader cannot check."""
    retriever, _ = _retriever(RANKED)

    text = retriever._retrieve_ranked_candidates(
        QueryRoute(route="structured", ranking="tenure")
    ).text

    assert "11y as Staff Engineer at Steady Ltd (2015-present)" in text


def test_an_experience_question_orders_on_the_career_column() -> None:
    retriever, connection = _retriever(RANKED)

    retriever._retrieve_ranked_candidates(QueryRoute(route="structured", ranking="experience"))

    assert "ORDER BY years_experience DESC NULLS LAST" in connection.statements[0]


def test_a_skill_ranking_says_the_years_are_not_years_of_that_skill() -> None:
    """The CVs date jobs, not skills — the answer must not claim more than that."""
    retriever, connection = _retriever(RANKED)

    context = retriever._retrieve_ranked_candidates(
        QueryRoute(route="structured", skills=["Python"], ranking="experience")
    )

    assert "unnest(skills)" in connection.statements[0]
    assert "not years spent using any one of them" in context.text
    assert "whose CVs list Python" in context.text


def test_a_ranking_over_nothing_says_so_rather_than_ranking_nothing() -> None:
    retriever, _ = _retriever([])

    context = retriever._retrieve_ranked_candidates(
        QueryRoute(route="structured", skills=["Cobol"], ranking="tenure")
    )

    assert context.source_files == []
    assert "No candidate" in context.text


def test_a_breakdown_sends_what_people_do_and_not_where_they_studied() -> None:
    """Thirty rows at once means every column is paid for thirty times."""
    retriever, connection = _retriever([("ada.pdf", "Ada Lovelace", "Engineer", ["Python"])])

    retriever._retrieve_all_candidates_matching_filters(
        QueryRoute(route="structured", breakdown=True)
    )

    selected = connection.statements[0].split("FROM")[0]
    assert "role" in selected and "skills" in selected
    assert "institutions" not in selected and "current_company" not in selected
