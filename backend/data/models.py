from dataclasses import dataclass, field

from pydantic import BaseModel

ROUTES: tuple[str, ...] = ("structured", "profile", "semantic")

# Superlatives the structured route can settle by ordering rows rather than by
# reading prose: how long a career runs, and how long one job lasted.
RANKINGS: tuple[str, ...] = ("experience", "tenure")

CandidateRow = tuple[str, str, str | None, str | None, int | None, list[str], list[str]]
ChunkRow = tuple[str, str, str | None, str]
NameMatchRow = tuple[int, str, str, str | None, str | None]


class ChatRequest(BaseModel):
    question: str
    route: str | None = None


@dataclass
class CvChunk:
    section: str | None
    content: str


@dataclass
class QueryRoute:
    route: str
    candidate_name: str | None = None
    skills: list[str] = field(default_factory=list)
    institution: str | None = None
    minimum_years_experience: int | None = None
    # One of RANKINGS when the question asks who has the most of something. A
    # filter says which rows qualify; this says what makes one of them the
    # answer.
    ranking: str | None = None
    # Set when the question asks how the corpus divides up rather than who
    # matches. The answer needs every candidate and little of each: what they
    # do, not where they studied.
    breakdown: bool = False


@dataclass
class RetrievedContext:
    text: str
    source_files: list[str]
    disambiguation_note: str | None = None
