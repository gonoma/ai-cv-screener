"""What happens when a model answers a JSON request with something else.

`json.loads` reports the *position* of the problem and nothing about its cause —
"Expecting value: line 1 column 1 (char 0)" is the same message for an empty
string, a refusal and a code fence. After a batch that took sixteen minutes to
fail, that is a poor thing to be left holding.
"""

import pytest

from backend.providers.llm_provider import LlmProvider


def test_plain_json_parses() -> None:
    assert LlmProvider._parse_json(
        text='{"candidates": []}',
        model_name="m",
    ) == {"candidates": []}


@pytest.mark.parametrize(
    "wrapped",
    ['```json\n{"a": 1}\n```', '```\n{"a": 1}\n```', '  ```JSON\n{"a": 1}\n```  '],
)
def test_a_markdown_code_fence_is_unwrapped_rather_than_rejected(wrapped: str) -> None:
    """The model did as asked in the wrong wrapper; unwrapping beats re-asking."""
    assert LlmProvider._parse_json(
        text=wrapped,
        model_name="m",
    ) == {"a": 1}


@pytest.mark.parametrize("empty", ["", "   ", "\n\n", None])
def test_an_empty_response_says_so(empty) -> None:
    with pytest.raises(RuntimeError, match="empty response"):
        LlmProvider._parse_json(
            text=empty,
            model_name="nemotron",
        )


def test_prose_instead_of_json_quotes_what_arrived() -> None:
    """The text is the whole diagnosis — a refusal and a truncation need different fixes."""
    with pytest.raises(RuntimeError, match="I cannot help"):
        LlmProvider._parse_json(
            text="I cannot help with that.",
            model_name="nemotron",
        )


def test_truncated_json_quotes_what_arrived() -> None:
    with pytest.raises(RuntimeError, match="not JSON"):
        LlmProvider._parse_json(
            text='{"candidates": [{"name": "Ada',
            model_name="nemotron",
        )


def test_the_quoted_text_is_trimmed() -> None:
    """A chat bubble and a log line are not the place for a whole response body."""
    with pytest.raises(RuntimeError) as raised:
        LlmProvider._parse_json(
            text="x" * 5000,
            model_name="m",
        )
    assert len(str(raised.value)) < 300


def test_the_model_is_named_so_the_report_is_actionable() -> None:
    with pytest.raises(RuntimeError, match="nvidia/nemotron"):
        LlmProvider._parse_json(
            text="nope",
            model_name="nvidia/nemotron",
        )
