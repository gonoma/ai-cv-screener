import random
from dataclasses import dataclass

_ROLES: tuple[str, ...] = (
    "backend engineer",
    "frontend engineer",
    "mobile engineer",
    "data engineer",
    "machine learning engineer",
    "site reliability engineer",
    "platform engineer",
    "QA engineer",
    "security engineer",
    "product manager",
    "UX designer",
    "technical writer",
    "engineering manager",
    "data analyst",
    "solutions architect",
)

_SENIORITIES: tuple[str, ...] = (
    "intern",
    "junior",
    "mid-level",
    "senior",
    "staff",
    "principal",
    "lead",
    "director",
)

_CITIES: tuple[str, ...] = (
    "Barcelona, Spain",
    "Madrid, Spain",
    "Valencia, Spain",
    "Seville, Spain",
    "Bilbao, Spain",
    "Zaragoza, Spain",
    "Malaga, Spain",
    "Lisbon, Portugal",
    "Porto, Portugal",
    "Milan, Italy",
    "Berlin, Germany",
    "Munich, Germany",
    "Amsterdam, Netherlands",
    "Delft, Netherlands",
    "Paris, France",
    "Lyon, France",
    "Dublin, Ireland",
    "Stockholm, Sweden",
    "Copenhagen, Denmark",
    "Warsaw, Poland",
    "Krakow, Poland",
    "Bucharest, Romania",
    "Prague, Czechia",
    "Vienna, Austria",
    "Tallinn, Estonia",
    "Casablanca, Morocco",
    "Bogota, Colombia",
    "Buenos Aires, Argentina",
    "Mexico City, Mexico",
    "Bangalore, India",
    "Toronto, Canada",
)


_EDUCATION: tuple[str, ...] = (
    "Universitat Politecnica de Catalunya (UPC)",
    "Universidad Politecnica de Madrid (UPM)",
    "Universitat de Barcelona (UB)",
    "Universitat Autonoma de Barcelona (UAB)",
    "Universidad de Sevilla",
    "Universidad de Zaragoza",
    "Universidad Carlos III de Madrid",
    "Instituto Superior Tecnico, Lisbon",
    "TU Delft",
    "KTH Royal Institute of Technology",
    "Politecnico di Milano",
    "Trinity College Dublin",
    "TU Munich",
    "Sorbonne Universite",
    "AGH University of Science and Technology",
    "Universitatea Politehnica din Bucuresti",
    "a coding bootcamp, no university degree",
    "self-taught, no formal qualification",
    "a PhD in a quantitative field",
    "a vocational qualification, no university degree",
)

_INDUSTRIES: tuple[str, ...] = (
    "banking",
    "insurance",
    "e-commerce",
    "travel booking",
    "logistics",
    "healthtech",
    "biotech",
    "govtech",
    "energy",
    "telecoms",
    "gaming",
    "adtech",
    "edtech",
    "agritech",
    "media streaming",
    "cybersecurity products",
    "industrial automation",
    "consultancy",
)

_CAREER_SHAPES: tuple[str, ...] = (
    "a linear climber at one or two employers",
    "a career changer arriving from teaching",
    "a career changer arriving from finance",
    "a career changer arriving from hospitality",
    "a contractor with several short posts",
    "someone returning after a multi-year gap",
    "a long tenure of eight or more years at one employer",
    "a freelancer mixing part-time clients",
    "someone who moved country mid-career",
    "someone promoted internally three times at one company",
)

_LANGUAGES: tuple[str, ...] = (
    "Spanish",
    "Catalan",
    "Basque",
    "Galician",
    "Portuguese",
    "Italian",
    "French",
    "German",
    "Polish",
    "Romanian",
    "Dutch",
    "Swedish",
    "Arabic",
    "Hindi",
    "Mandarin",
    "English",
)

_SEED: int = 0


@dataclass(frozen=True)
class CandidateSpec:
    """One Candidate's specs, fixed before any LLM call sees them."""

    role: str
    seniority: str
    city: str
    years_experience: int
    education: str
    industry: str
    career_shape: str
    primary_language: str

    def as_brief(self) -> str:
        years = "1 year" if self.years_experience == 1 else f"{self.years_experience} years"
        return (
            f"{self.seniority} {self.role} in {self.city}, "
            f"{years} of experience, "
            f"educated at {self.education}, working in {self.industry}. "
            f"Career shape: {self.career_shape}. "
            f"First language {self.primary_language}."
        )


def build_spec_matrix(count: int) -> list[CandidateSpec]:
    """Combine the axes without replacement, in code, before any model is called.

    This is what stops the corpus collapsing onto one archetype. A model converges
    when asked to invent people (same mid-level full-stack engineer with same 8 skills),
    and it converges harder inside a single response than across separate ones.
    By fixing the coordinates here, we turn each call from "invent five people" to
    "write five CVs for these five people", which is a narrower task with less room
    to drift.

    Seeded, so a given corpus size always yields the same matrix: a failing eval/test is
    then a real regression rather than a reshuffle.

    Coordinates are dealt, not sampled, so the corpus cannot pile five people into
    one city while a dozen others go unused.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")

    rng = random.Random(_SEED)
    roles = _draw_without_replacement(_ROLES, count, rng)
    seniorities = _draw_without_replacement(_SENIORITIES, count, rng)
    cities = _draw_without_replacement(_CITIES, count, rng)
    education = _draw_without_replacement(_EDUCATION, count, rng)
    industries = _draw_without_replacement(_INDUSTRIES, count, rng)
    shapes = _draw_without_replacement(_CAREER_SHAPES, count, rng)
    languages = _draw_without_replacement(_LANGUAGES, count, rng)

    return [
        CandidateSpec(
            role=roles[index],
            seniority=seniorities[index],
            city=cities[index],
            years_experience=_years_for(seniorities[index], rng),
            education=education[index],
            industry=industries[index],
            career_shape=shapes[index],
            primary_language=languages[index],
        )
        for index in range(count)
    ]


def _draw_without_replacement(pool: tuple[str, ...], count: int, rng: random.Random) -> list[str]:
    drawn: list[str] = []
    while len(drawn) < count:
        deck = list(pool)
        rng.shuffle(deck)
        drawn.extend(deck)
    return drawn[:count]


def _years_for(seniority: str, rng: random.Random) -> int:
    spans = {
        "intern": (0, 1),
        "junior": (1, 3),
        "mid-level": (3, 6),
        "senior": (6, 11),
        "lead": (8, 14),
        "staff": (9, 15),
        "principal": (12, 20),
        "director": (13, 22),
    }
    low, high = spans[seniority]
    return rng.randint(low, high)
