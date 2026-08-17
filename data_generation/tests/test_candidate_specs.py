from collections import Counter

from data_generation.models.candidate_specs import build_spec_matrix

CORPUS_SIZE = 30

# Each axis draws without replacement, so it may only repeat ceil(count / pool)
# times. A shrunk pool should fail here rather than quietly clustering the
# corpus — twenty of thirty candidates in Barcelona tests nothing about
# retrieval and would still pass a uniqueness check on whole records.
MAX_REPEATS = {
    "role": 2,
    "seniority": 4,
    "city": 1,
    "education": 2,
    "industry": 2,
    "career_shape": 3,
    "primary_language": 2,
}

SENIORITY_CEILINGS = {
    "intern": 2,
    "junior": 4,
    "mid-level": 7,
    "senior": 12,
    "lead": 15,
    "staff": 16,
    "principal": 21,
    "director": 23,
}


def test_no_axis_clusters():
    specs = build_spec_matrix(CORPUS_SIZE)

    for axis, ceiling in MAX_REPEATS.items():
        values = Counter(getattr(spec, axis) for spec in specs)
        assert max(values.values()) <= ceiling, f"{axis} clustered: {values.most_common(3)}"


def test_seniority_and_experience_never_contradict():
    """A director with two years would make the multi-hop eval grade nonsense."""
    for spec in build_spec_matrix(CORPUS_SIZE):
        assert spec.years_experience <= SENIORITY_CEILINGS[spec.seniority]


def test_matrix_is_reproducible():
    assert build_spec_matrix(CORPUS_SIZE) == build_spec_matrix(CORPUS_SIZE)
