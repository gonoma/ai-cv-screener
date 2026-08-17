import itertools
import json
import pathlib
import re
from collections import Counter

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GROUND_TRUTH = REPO_ROOT / "data" / "ground_truth.json"

if not GROUND_TRUTH.exists():
    pytest.skip("no generated corpus", allow_module_level=True)

RECORDS = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))["candidates"]

# Measured over the committed corpus: the most similar pair sits at 0.28 and the
# median at 0.13. 0.45 is roughly 60% above the observed worst case, so ordinary
# overlap between two backend engineers passes, while the failure this exists to
# catch does not — a model that has settled into a template produces pairs well
# above 0.5, because the phrasing and structure repeat rather than just the
# vocabulary.
MAX_PAIRWISE_SIMILARITY = 0.45


def candidate_text(record: dict) -> str:
    """
    The CV body, minus name, email and phone.

    Those are unique by construction, so they would lower every pair's score
    evenly and mask the body text that reveals a template.
    """
    parts = [record["summary"], " ".join(record["skills"])]
    for role in record["experience"]:
        parts += [role["company"], role["role"], " ".join(role["bullets"])]
    for qualification in record["education"]:
        parts += [qualification["institution"], qualification["degree"], qualification["field"]]
    return " ".join(parts)


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(left: str, right: str) -> float:
    first, second = tokens(left), tokens(right)
    return len(first & second) / len(first | second)


def test_emails_are_unique():
    """The email is the identifier, so it stays unique even when two names match."""
    emails = [record["email"].lower() for record in RECORDS]

    assert [email for email, count in Counter(emails).items() if count > 1] == []


def test_no_two_candidates_share_a_skill_set():
    skill_sets = [tuple(sorted(skill.lower() for skill in record["skills"])) for record in RECORDS]

    assert len(set(skill_sets)) == len(skill_sets)


def test_no_pair_of_candidates_reads_as_the_same_cv():
    texts = {record["id"]: candidate_text(record) for record in RECORDS}
    too_similar = [
        (left, right, round(score, 3))
        for (left, right) in itertools.combinations(sorted(texts), 2)
        if (score := jaccard(texts[left], texts[right])) > MAX_PAIRWISE_SIMILARITY
    ]

    assert too_similar == []
