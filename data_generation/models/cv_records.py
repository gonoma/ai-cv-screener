from typing import Any

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
    summary: str = Field(description="Two or three sentences, first person omitted.")
    skills: list[str] = Field(description="Six to fourteen concrete technical skills.")
    languages: list[str] = Field(
        description="Spoken languages with level, e.g. 'Spanish (native)'."
    )
    experience: list[Role] = Field(description="Most recent role first.")
    education: list[Qualification]

    def current_role(self) -> Role:
        return self.experience[0]

    def years_of_experience(self) -> int:
        earliest_start = min(role.start_year for role in self.experience)
        latest_end = max(role.end_year_or_present() for role in self.experience)
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


class CandidateBatch(BaseModel):
    candidates: list[Candidate] = Field(
        description="One record per brief, in the same order as the briefs given."
    )
