"""The numbers a CV implies rather than states, derived here instead of asked for.

Years of experience and length of tenure are arithmetic over dates the CV
prints. Asking the model for the arithmetic put a wrong number in the row and
no way to tell — one candidate came back with three years where the dates say
four, and nothing downstream could disagree with it. Asking for the dates and
doing the sums here makes the same numbers reproducible, free, and testable
against the answer key.
"""

from datetime import date
from typing import Any

Position = dict[str, Any]


def current_year() -> int:
    return date.today().year


def career_years(positions: list[Position], as_of: int | None = None) -> int | None:
    """Earliest start to latest end, a current role ending this year.

    The span, not the sum of the parts: two jobs held in the same years are one
    stretch of a career, and adding them would credit the person twice.

    Rows that name no job are left out, the same as in `longest_tenure`. One CV
    in the corpus carries a 1995-2006 entry whose role and employer are both the
    literal string "None" — years in which its owner was a child — and counting
    it made a candidate with a 20-year career rank first on 31.
    """
    dated = [position for position in positions if _names_a_job(position)]
    starts = [start for position in dated if (start := _year(position.get("start_year")))]
    if not starts:
        return None
    return max(0, _latest_end(dated, as_of) - min(starts))


def longest_tenure(positions: list[Position], as_of: int | None = None) -> Position | None:
    """The single position held longest, which is not the same as the longest career.

    Only positions that name a job: the corpus contains rows whose role and
    employer are both the literal string "None", spanning years nobody worked,
    and one of them is long enough to top the ranking. A row that cannot be
    described is not an answer to "who stayed longest in one job" — it is a hole
    in a CV, and citing it would put "None at None" in front of a recruiter.

    Ties go to the earlier entry, which is how the CV itself orders them.
    """
    dated = [
        position
        for position in positions
        if _year(position.get("start_year")) and _names_a_job(position)
    ]
    if not dated:
        return None
    return max(dated, key=lambda position: tenure_years(position, as_of))


def tenure_years(position: Position, as_of: int | None = None) -> int:
    """How long one position lasted, counting an open end as still running.

    A role that starts and ends in the same year is a year of someone's life,
    not zero — otherwise a string of short contracts scores as no experience at
    all. So the floor is one, and the arithmetic only exceeds it on a genuine
    multi-year role.
    """
    start = _year(position.get("start_year"))
    if start is None:
        return 0
    end = _year(position.get("end_year")) or (as_of or current_year())
    return max(1, end - start)


def current_position(positions: list[Position]) -> Position | None:
    """The job the person holds now: an open end if there is one, else the latest.

    Read off the dates rather than taken as the first entry, because CVs are not
    all reverse-chronological and a sidebar layout can hand the parser the
    oldest role first.
    """
    named = [position for position in positions if _names_a_job(position)]
    if not named:
        return None
    open_ended = [position for position in named if _year(position.get("end_year")) is None]
    if open_ended:
        return max(open_ended, key=lambda position: _year(position.get("start_year")) or 0)
    return max(named, key=lambda position: _year(position.get("end_year")) or 0)


# What a model writes where a CV printed nothing. "None" is a string here, not a
# missing value: it survived generation, rendering and extraction as text.
_UNNAMED: frozenset[str] = frozenset({"", "none", "null", "n/a", "na", "-", "unknown", "?"})


def _names_a_job(position: Position) -> bool:
    return any(
        str(position.get(field) or "").strip().lower() not in _UNNAMED
        for field in ("role", "company")
    )


def _latest_end(positions: list[Position], as_of: int | None = None) -> int:
    this_year = as_of or current_year()
    ends = [
        this_year if _year(position.get("end_year")) is None else _year(position["end_year"])
        for position in positions
        if _year(position.get("start_year"))
    ]
    return max(ends) if ends else this_year


def _year(value: Any) -> int | None:
    """Accept the integer the schema asks for and the "2019" a model sends anyway."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip()[:4])
    except (ValueError, TypeError):
        return None


# " - ", ", ", " @ " and friends: how a model glues an employer onto a job title
# when the CV prints them on the same line.
_EMPLOYER_SEPARATORS: tuple[str, ...] = (" - ", " – ", " — ", ", ", " @ ", " at ")


def role_without_company(role: str | None, company: str | None) -> str | None:
    """Drop an employer the model appended to the job title.

    "Lead Engineering Manager - SummitEd Inc." is the same role as "Lead
    Engineering Manager", but it does not read as one in a roster line, and the
    employer is already its own column. Only a suffix that matches the company
    on the same record is removed, so a title that genuinely contains a comma
    survives.
    """
    if not role or not company:
        return role
    for separator in _EMPLOYER_SEPARATORS:
        suffix = f"{separator}{company}"
        if role.lower().endswith(suffix.lower()):
            return role[: -len(suffix)].strip(" ,-–—")
    return role
