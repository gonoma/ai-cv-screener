"""Whether the answer is asked to cite, which depends on there being a choice.

A citation says *which* source a claim came from. With one source there is
nothing to say, and repeating the same filename after every sentence is noise
the reader has to look past — the UI already names the document underneath.
"""

from backend.data.models import RetrievedContext
from backend.domain.query import AnswerGenerator

GENERATOR = AnswerGenerator()


def _prompt(*source_files: str) -> str:
    return GENERATOR._system_prompt(RetrievedContext(text="ctx", source_files=list(source_files)))


def test_one_source_is_not_cited() -> None:
    prompt = _prompt("ana-silva.pdf")
    assert "do not cite" in prompt
    assert "Cite the source file" not in prompt


def test_the_same_source_repeated_is_still_one_source() -> None:
    """Chunks of one CV arrive as many rows naming the same file."""
    assert "do not cite" in _prompt("ana-silva.pdf", "ana-silva.pdf", "ana-silva.pdf")


def test_several_sources_are_cited() -> None:
    prompt = _prompt("ana-silva.pdf", "jana-novak.pdf")
    assert "Cite the source file" in prompt
    assert "do not cite" not in prompt


def test_two_people_sharing_a_name_are_cited() -> None:
    """The profile route's ambiguous case: one name, two files, citations essential."""
    assert "Cite the source file" in _prompt("ana-silva.pdf", "ana-silva-2.pdf")


def test_no_sources_at_all_asks_for_no_citations() -> None:
    """ "Nothing matched that filter" has nothing to cite."""
    assert "do not cite" in _prompt()


def test_the_grounding_rules_survive_either_way() -> None:
    """The citation switch must not disturb what stops the model inventing CVs."""
    for prompt in (_prompt("a.pdf"), _prompt("a.pdf", "b.pdf")):
        assert "no general knowledge" in prompt
        assert "name all of them" in prompt
