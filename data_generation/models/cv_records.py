from typing import Any, Literal

from pydantic import BaseModel, Field

CORPUS_YEAR = 2026


class Role(BaseModel):
    company: str
    role: str
    location: str
    start_year: int
    end_year: int | None = Field(
        description="Year the role ended, or null if this is the current role."
    )
    bullets: list[str] = Field(description="Two to four achievement bullets.")

    def end_year_or_present(self) -> int:
        return self.end_year or CORPUS_YEAR

    def names_a_job(self) -> bool:
        """Whether this row is a job at all.

        The model that writes these records occasionally emits a filler row whose
        company and role are both the string "None", spanning years its owner was
        a child. Counted, it added eleven years to one candidate's career.
        """
        return any(field.strip().lower() not in ("", "none") for field in (self.role, self.company))


class Qualification(BaseModel):
    institution: str = Field(
        description="Full institution name, e.g. 'Universitat Politecnica de Catalunya'."
    )
    short_name: str = Field(
        description="How the institution is commonly abbreviated, e.g. 'UPC'. "
        "Repeat the full name if it has no common abbreviation."
    )
    degree: str
    field: str
    graduation_year: int

    def both_names(self) -> set[str]:
        return {self.institution, self.short_name}


class Candidate(BaseModel):
    name: str
    headline: str = Field(description="Current job title, e.g. 'Senior Data Engineer'.")
    email: str
    phone: str
    location: str = Field(description="City, Country.")
    presentation: Literal["a woman", "a man", "a non-binary person"] = Field(
        description="How this person presents in a photograph. Must be consistent "
        "with the given name. Vary this across the corpus."
    )
    appearance: str = Field(
        description="One short phrase for a headshot, consistent with the name and "
        "career length: approximate age, build, skin tone, hair, and one ordinary "
        "distinguishing feature. For example 'in her late forties, heavy build, "
        "olive skin, short greying hair, wire-rimmed glasses'. No clothing, no "
        "background."
    )
    summary: str = Field(description="Two or three sentences, first person omitted.")
    skills: list[str] = Field(description="Six to fourteen concrete technical skills.")
    languages: list[str] = Field(
        description="Spoken languages with level, e.g. 'Spanish (native)'."
    )
    experience: list[Role] = Field(description="Most recent role first.")
    education: list[Qualification]

    def current_role(self) -> Role:
        """The job held now, read off the dates rather than taken as the first entry.

        `experience` is documented as most-recent-first and usually is, but the
        model that writes these records sometimes lists a career oldest-first —
        and then the answer key called someone's first job out of university
        their current role. A wrong answer key is worse than none: it grades a
        correct extraction as a failure.
        """
        current = [role for role in self.experience if role.end_year is None]
        if current:
            return max(current, key=lambda role: role.start_year)
        return max(self.experience, key=lambda role: role.end_year_or_present())

    def years_of_experience(self) -> int:
        jobs = [role for role in self.experience if role.names_a_job()] or self.experience
        earliest_start = min(role.start_year for role in jobs)
        latest_end = max(role.end_year_or_present() for role in jobs)
        return latest_end - earliest_start

    def institution_names(self) -> list[str]:
        return sorted(
            {name for qualification in self.education for name in qualification.both_names()}
        )

    def to_ground_truth_record(self, cv_id: str, source_file: str, template: str) -> dict[str, Any]:
        current = self.current_role()
        return {
            "id": cv_id,
            "source_file": source_file,
            "template": template,
            "current_role": current.role,
            "current_company": current.company,
            "years_experience": self.years_of_experience(),
            "institutions": self.institution_names(),
            **self.model_dump(),
        }

    @classmethod
    def from_ground_truth_record(cls, entry: dict[str, Any]) -> "Candidate":
        return cls.model_validate(entry)
