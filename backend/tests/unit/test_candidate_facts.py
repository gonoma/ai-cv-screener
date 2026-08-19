"""The arithmetic over a CV's dates, which used to be asked of the model.

A model that returns 3 where the dates say 4 is not obviously wrong to anything
downstream — the row is well-formed and the number is plausible. These are the
cases that made asking a bad trade.
"""

from backend.domain import candidate_facts

CORPUS_YEAR = 2026


def _position(start: int, end: int | None, role: str = "Engineer", company: str = "Acme") -> dict:
    return {"role": role, "company": company, "start_year": start, "end_year": end}


def test_a_career_is_measured_end_to_end_not_added_up() -> None:
    """Two roles held over the same years are one stretch of a career, not two."""
    positions = [_position(2019, 2021), _position(2021, None)]

    assert candidate_facts.career_years(positions, as_of=CORPUS_YEAR) == 7


def test_a_current_role_runs_to_this_year() -> None:
    assert candidate_facts.career_years([_position(2022, None)], as_of=CORPUS_YEAR) == 4


def test_a_candidate_with_no_dated_positions_has_no_number() -> None:
    """No answer beats a wrong one: the column stays null and the row says nothing."""
    assert candidate_facts.career_years([]) is None
    assert candidate_facts.career_years([{"role": "Engineer", "company": "Acme"}]) is None


def test_the_longest_position_is_not_the_longest_career() -> None:
    positions = [_position(2015, 2026, company="Steady Ltd"), _position(2012, 2015)]

    longest = candidate_facts.longest_tenure(positions, as_of=CORPUS_YEAR)

    assert longest["company"] == "Steady Ltd"
    assert candidate_facts.tenure_years(longest, as_of=CORPUS_YEAR) == 11


def test_a_role_that_starts_and_ends_in_one_year_still_counts_as_a_year() -> None:
    """Otherwise a run of short contracts ranks as no experience at all."""
    assert candidate_facts.tenure_years(_position(2022, 2022), as_of=CORPUS_YEAR) == 1


def test_the_current_position_is_the_open_ended_one_wherever_it_sits() -> None:
    """CVs are not all reverse-chronological, and a sidebar layout can invert them."""
    positions = [_position(2018, 2021, role="Junior"), _position(2021, None, role="Staff")]

    assert candidate_facts.current_position(positions)["role"] == "Staff"


def test_with_no_current_role_the_latest_one_stands_in() -> None:
    positions = [_position(2010, 2014, role="Analyst"), _position(2014, 2019, role="Lead")]

    assert candidate_facts.current_position(positions)["role"] == "Lead"


def test_a_year_sent_as_text_is_still_a_year() -> None:
    """Schemas ask for integers; models send "2019" anyway, and a crash here fails an ingest."""
    positions = [{"role": "Engineer", "company": "Acme", "start_year": "2019", "end_year": None}]

    assert candidate_facts.career_years(positions, as_of=CORPUS_YEAR) == 7


def test_an_employer_glued_to_the_job_title_is_removed() -> None:
    """The employer is already its own column; in the title it is noise in every roster line."""
    glued = "Lead Engineering Manager - SummitEd Inc."
    stripped = candidate_facts.role_without_company(glued, "SummitEd Inc.")

    assert stripped == "Lead Engineering Manager"
    assert (
        candidate_facts.role_without_company("Data Consultant, Self-employed", "Self-employed")
        == "Data Consultant"
    )


def test_a_title_that_merely_contains_a_comma_is_left_alone() -> None:
    """Only a suffix matching this record's own employer goes."""
    assert (
        candidate_facts.role_without_company("Director, Platform Engineering", "Vodafone")
        == "Director, Platform Engineering"
    )


def test_a_position_naming_no_job_cannot_win_the_ranking() -> None:
    """The corpus contains rows whose role and employer are both the string "None"."""
    positions = [
        {"role": "None", "company": "None", "start_year": 1995, "end_year": 2006},
        {"role": "Support Specialist", "company": "KPN", "start_year": 2016, "end_year": 2024},
    ]

    longest = candidate_facts.longest_tenure(positions, as_of=CORPUS_YEAR)

    assert longest["company"] == "KPN"


def test_a_nameless_row_is_not_mistaken_for_the_current_job() -> None:
    positions = [
        {"role": "Staff Engineer", "company": "Acme", "start_year": 2021, "end_year": 2024},
        {"role": "", "company": None, "start_year": 2025, "end_year": None},
    ]

    assert candidate_facts.current_position(positions)["company"] == "Acme"


def test_a_career_does_not_start_in_a_year_naming_no_job() -> None:
    """A "None at None" row spanning someone's childhood used to add eleven years to it."""
    positions = [
        {"role": "None", "company": "None", "start_year": 1995, "end_year": 2006},
        {"role": "Support Specialist", "company": "KPN", "start_year": 2016, "end_year": None},
    ]

    assert candidate_facts.career_years(positions, as_of=CORPUS_YEAR) == 10
