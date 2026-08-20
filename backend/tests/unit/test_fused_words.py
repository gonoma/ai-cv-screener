"""Words run together by pypdf, and the ones that only look run together.

A two-column skills block comes back from pypdf as "Grafana ELK StackPython",
and a skill fused to its neighbour is invisible to the exact-match filter that
answers "who knows Python" — one candidate in thirty, silently absent from a
route whose whole promise is returning every match.

The repair borrows word boundaries from pdfplumber, which reads the columns in a
worse order but does put the spaces in. What these cases pin down is the other
half: that it leaves alone the technologies which are spelled with a capital in
the middle on purpose.
"""

from backend.domain.ingestion import PdfTextExtractor

REPAIR = PdfTextExtractor._split_run_together_words
WORDS = PdfTextExtractor._words


def test_two_skills_run_together_are_separated() -> None:
    reference = "Grafana ELK Stack Python Bash AWS"

    assert REPAIR("Grafana ELK StackPython", WORDS(reference)) == "Grafana ELK Stack Python"


def test_a_name_fused_to_a_headline_is_separated() -> None:
    """The commonest case: the sidebar templates butt the name against the job title."""
    reference = "Alejandro Martínez Data Engineer"

    assert REPAIR("Alejandro MartínezData", WORDS(reference)) == "Alejandro Martínez Data"


def test_a_technology_spelled_with_a_capital_inside_is_left_alone() -> None:
    """Both readings agree it is one word, so it is one word."""
    reference = "JavaScript TypeScript PostgreSQL GitHub"

    assert REPAIR("JavaScript PostgreSQL", WORDS(reference)) == "JavaScript PostgreSQL"


def test_a_fused_word_survives_when_only_one_half_is_a_word() -> None:
    """ "PyTorch" must not become "Py Torch" because the page happens to mention Torch."""
    reference = "Torch models trained end to end"

    assert REPAIR("PyTorch", WORDS(reference)) == "PyTorch"


def test_a_camel_case_word_the_reference_never_saw_is_left_alone() -> None:
    """With nothing to check against, the safe move is to change nothing."""
    assert REPAIR("StackPython", set()) == "StackPython"
    assert REPAIR("StackPython", WORDS("some unrelated page")) == "StackPython"


def test_the_rest_of_the_line_is_untouched() -> None:
    """Punctuation and spacing are load-bearing: the chunker finds headings by line."""
    reference = "SKILLS Kubernetes Docker CI/CD (GitLab CI)"

    repaired = REPAIR("SKILLS\nKubernetesDocker\nCI/CD (GitLab CI)", WORDS(reference))

    assert repaired == "SKILLS\nKubernetes Docker\nCI/CD (GitLab CI)"
