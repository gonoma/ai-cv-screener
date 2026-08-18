"""The failure text a recruiter actually reads.

Each case here is a failure with a *different answer*: waiting fixes one,
editing `.env` fixes another, and nothing the user can do fixes the rest. A
message that does not distinguish them sends people to the wrong place.
"""

import pytest

from backend.endpoints.chat import _readable


@pytest.mark.parametrize(
    "failure,expected",
    [
        (Exception("Error code: 429 - quota exceeded"), "rate limiting"),
        (Exception("RESOURCE_EXHAUSTED"), "rate limiting"),
        (Exception("Error code: 401 - bad key"), "API key"),
        (Exception("invalid api key supplied"), "API key"),
        (Exception("Request timed out"), "too long"),
        (Exception("Connection error"), "Could not reach"),
        (RuntimeError("model returned no content"), "empty answer"),
    ],
)
def test_known_failures_are_named(failure: Exception, expected: str) -> None:
    assert expected in _readable(failure)


def test_a_rate_limit_carries_the_wait_when_the_provider_gives_one() -> None:
    message = _readable(Exception("429. Please retry in 21.54276431s."))
    assert "about 22s" in message


def test_a_rate_limit_without_a_wait_still_reads_cleanly() -> None:
    message = _readable(Exception("Error code: 429 too_many_requests"))
    assert message.endswith("rate limiting this key.")


def test_an_unknown_failure_keeps_its_own_text() -> None:
    """Flattening this into "something went wrong" would delete the only clue."""
    message = _readable(ValueError("chunk 7 has no embedding\nstack frame here"))
    assert message == "Unknown error: ValueError: chunk 7 has no embedding"


def test_an_unknown_failure_is_trimmed_rather_than_pasted_whole() -> None:
    assert len(_readable(ValueError("x" * 900))) < 340
