import json
import pathlib
import re

import pytest

from backend.domain.ingestion import PdfTextExtractor

pytestmark = pytest.mark.integration

EXTRACTOR = PdfTextExtractor()

ROOT = pathlib.Path(__file__).resolve().parents[3]
KEY = ROOT / "data" / "ground_truth.json"

if not KEY.exists():
    pytest.skip("no generated corpus", allow_module_level=True)

RECORDS = json.loads(KEY.read_text())["candidates"]

# One CV per template variant, not all thirty. Layout is the variable that
# breaks extraction; a second CV through the same template exercises the same
# code path and only slows the suite down.
PER_TEMPLATE = list({record["template"]: record for record in RECORDS}.values())

# Measured across the templates: most CVs keep every skill, and the worst loses
# one of twelve to a line break inside a compound term ("Linux/bash"). 0.85
# tolerates that while still failing the case that matters — a skills block that
# extracts as an unreadable column, dropping the fraction to nearly zero.
MINIMUM_SKILL_SURVIVAL = 0.85


def flatten(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def extracted(record) -> str:
    return flatten(EXTRACTOR.extract_text_from_pdf(ROOT / "data" / "cvs" / record["source_file"]))


@pytest.mark.parametrize("record", PER_TEMPLATE, ids=lambda r: r["template"])
def test_expected_strings_survive_extraction(record):
    text = extracted(record)

    assert record["name"].split()[-1] in text, "surname missing"
    assert record["email"] in text, "email missing"
    assert len(text.split()) > 150, f"only {len(text.split())} words"

    for institution in record["institutions"]:
        assert flatten(institution) in text, f"institution {institution!r} missing"


@pytest.mark.parametrize("record", PER_TEMPLATE, ids=lambda r: r["template"])
def test_skills_survive_extraction(record):
    """Skills specifically, because §4.2 makes them load-bearing.

    The structured route answers "who knows Python" from the skills column, so a
    skill lost between the PDF and the extractor is invisible to SQL — and
    invisible in a way no aggregation eval can distinguish from a candidate who
    genuinely lacks it. Every other assertion here would still pass.
    """
    text = extracted(record)
    present = [skill for skill in record["skills"] if flatten(skill) in text]

    survival = len(present) / len(record["skills"])
    assert survival >= MINIMUM_SKILL_SURVIVAL, (
        f"{record['template']} lost {sorted(set(record['skills']) - set(present))}"
    )
