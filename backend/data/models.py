from dataclasses import dataclass, field

from pydantic import BaseModel

ROUTES: tuple[str, ...] = ("structured", "profile", "semantic")

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


@dataclass
class RetrievedContext:
    text: str
    source_files: list[str]
    disambiguation_note: str | None = None
